import os
from netCDF4 import Dataset, date2num, num2date
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile
import numpy as np
from pygnssr.sim_gnssr.SimDataCube import SimDataCube, get_l1_vars_template
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.common.utils.gdalport import read_tiff, write_tiff, gen_gdal_ct, call_gdal_util
from datetime import datetime, timedelta
import click
import logging
from functools import partial
import multiprocessing as mp
import json
import glob
import subprocess
import gdal
from time import sleep
import shutil
import random
from matplotlib import pyplot as plt

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _cal_var_count(sat_list, rx_sat_arr, tx_sat_arr, var, tx_mask_list=None):
    sat_val = np.isin(rx_sat_arr, sat_list)
    var = np.ma.masked_where(~sat_val, var)
    if tx_mask_list is not None:
        tx_mask = np.isin(tx_sat_arr, tx_mask_list)
        var = np.ma.masked_where(tx_mask, var)
    return var.count(axis=0).astype('int32')


def cal_coverage_stat(ftile, sat_list, dir_sim_l1, dir_wb, tx_mask_list=None):

    dc = SimDataCube(ftile, "L1", os.path.join(dir_sim_l1, ftile.split('_')[0]), flag='r')
    # read sim variables

    rx_sat_arr = dc.nc.variables['sat_name'][:, :, :]
    tx_sat_arr = dc.nc.variables['tx_system'][:, :, :]

    var_sim = dc.nc.variables['sp_lon'][:, :, :]
    arr_count = _cal_var_count(sat_list, rx_sat_arr, tx_sat_arr, var_sim, tx_mask_list=tx_mask_list)

    # read water bodies data to apply water mask
    wb_file = os.path.join(dir_wb, ftile.split('_')[0], 'ESACCI-WB_' + ftile + '.tif')
    wb_arr, _ = read_tiff(wb_file)
    wb_arr = np.rot90(wb_arr, k=-1)
    # land indices
    val = np.broadcast_to(wb_arr == 1, arr_count.shape)
    arr_count_land = arr_count[val]

    dc.close_nc()
    return arr_count_land, arr_count


def plot_coverage_stat(ftile, arr_count_land, arr_count, dir_out):

    #file_png = os.path.join(dir_out, ftile+'_coverage.png')
    #plt.imshow(np.rot90(arr_count, k=1), vmin=0, vmax=10, cmap='Reds')
    #plt.savefig(file_png)
    #plt.close()

    tile_obj = Equi7Tile(ftile)
    dst_geotags = tile_obj.get_tile_geotags()
    file_tif = os.path.join(dir_out, ftile+'_coverage.tiff')
    write_tiff(file_tif, np.rot90(arr_count,  k=1), tiff_tags=dst_geotags)


    #------------------------------------
    xtick = 5
    bins = np.arange(xtick) - 0.5


    # calcualte cumulative histogram to find minimum number of observations with coverage above 95%
    N, bins, patches = plt.hist(arr_count_land, bins, density=True, cumulative=True)
    idx = N > 0.05
    min_count = np.where(N == np.min(N[idx]))[0][0]
    plt.close()

    N, bins, patches = plt.hist(arr_count_land, bins, density=True, histtype='bar',
                                facecolor='black', edgecolor='white')

    coverage = 100 - N[0]*100

    patches[min_count].set_facecolor('green')
    patches[0].set_facecolor('red')
    plt.title("Daily Coverage:  "+"{:.1f}".format(coverage)+" %   ("+str(min_count)+")", fontsize=22)
    plt.ylabel("Density", fontsize=22)
    plt.xlabel("Counts", fontsize=22)

    plt.ylim(0, 1)
    plt.yticks([0, 0.5, 1], fontsize=22)
    plt.xlim(-0.5, xtick)
    plt.xticks(fontsize=22)
    plt.tight_layout()
    file_png = os.path.join(dir_out, ftile+'_'+'_hist_land.png')
    plt.savefig(file_png)
    plt.close()

    return coverage, min_count


def plot_coverage_stat_wrapper(ftiles, sat_list, dir_sim_l1, dir_out, dir_wb, tx_mask_list=None):
    coverage_arr = []
    min_count_arr = []
    for ftile in ftiles:
        arr_count_land, arr_count = cal_coverage_stat(ftile, sat_list, dir_sim_l1, dir_wb, tx_mask_list=tx_mask_list)
        coverage, min_count = plot_coverage_stat(ftile, arr_count_land, arr_count, dir_out)
        coverage_arr.append(coverage)
        min_count_arr.append(min_count)

    return np.nanmean(coverage_arr), np.nanmean(min_count_arr)


def _cal_coverage(ftile, sat_list, dir_sim_l1, dir_wb, tx_mask_list=None):
    try:
        arr_count_land, arr_count = cal_coverage_stat(ftile, sat_list, dir_sim_l1, dir_wb, tx_mask_list=tx_mask_list)
        if len(arr_count_land) > 0:
            coverage = (np.count_nonzero(arr_count_land > 0) / len(arr_count_land))*100.0
        else:
            coverage = np.nan
        return coverage
    except OSError as e:
        print(ftile, e)
        logging.error("Failed to process:  " + ftile)
        return np.nan


def _cal_coverage_mean(ftiles, sat_list, dir_sim_l1, dir_wb, tx_mask_list=None, mp_num=1):
    if mp_num > 1:
        partial_func = partial(_cal_coverage, sat_list=sat_list, dir_sim_l1=dir_sim_l1, dir_wb=dir_wb, tx_mask_list=tx_mask_list)
        coverage_arr = mp.Pool(processes=mp_num).map(partial_func, ftiles)
    else:
        coverage_arr = []
        for ftile in ftiles:
            coverage_arr.append(_cal_coverage(ftile, sat_list, dir_sim_l1, dir_wb, tx_mask_list=tx_mask_list))

    coverage_mean = np.nanmean(coverage_arr)
    return coverage_mean


def find_optimal_constellation(ftiles, sat_list, dir_sim_l1, dir_wb, dir_out, date, sat_num, roi, tx_mask_list=None, mp_num=1):
    sat_list = np.array(sat_list)
    sat_groups = ["MIO", "EQO", "SSO06", "SSO08", "SSO10", "SSO12", "SSO14", "SSO16"]
    # orbit planes (10 orbit planes are defined)
    planes = [str(x) for x in range(10)]

    groups_arr = []
    coverage_achieved = False
    init_flag = True
    while not coverage_achieved:
        coverage_arr = []
        # initialize with Batch-2 launches
        sats_init = ["SPACEX-1", "SPACEX-5"]
        for group in groups_arr:
            r_plane = random.choice(planes)
            idx = np.char.find(sat_list, group+"_"+r_plane) != -1
            sats_init.extend(sat_list[idx])

        sats_arr = []
        if init_flag and len(sats_init) > 0:
            # run coverage calculation for initial satellites at the beginning
            coverage_arr.append(np.round(_cal_coverage_mean(ftiles, sats_init, dir_sim_l1, dir_wb,
                                                            tx_mask_list=tx_mask_list, mp_num=mp_num), decimals=1))
            sats_arr.append(sats_init)
        else:
            for i, sat_group in enumerate(sat_groups):
                r_plane = random.choice(planes)
                idx = np.char.find(sat_list, sat_group+"_"+r_plane) != -1
                sats_new = sat_list[idx]

                sats = np.append(sats_init,  sats_new)
                coverage_arr.append(np.round(_cal_coverage_mean(ftiles, sats, dir_sim_l1, dir_wb,
                                                                tx_mask_list=tx_mask_list, mp_num=mp_num), decimals=1))
                sats_arr.append(sats)

        max_ind = np.argmax(coverage_arr)
        print(coverage_arr)
        print('max:'+str(coverage_arr[max_ind]), sats_arr[max_ind])
        logging.info(coverage_arr)
        logging.info('max:'+str(coverage_arr[max_ind]))
        logging.info(sats_arr[max_ind])
        logging.info("------------------------------")

        if not init_flag:
            grp_idx = np.argmax(coverage_arr)
            groups_arr.extend([sat_groups[grp_idx]])

        if np.max(coverage_arr) > 95.0:
            coverage_achieved = True

        init_flag = False

    with open(os.path.join(dir_out, date+'_sim_'+str(sat_num)+'_sat_list_'+roi+'.txt'), 'w') as f:
        for item in sats_arr[max_ind]:
            f.write("{}, ".format(item))




def main():
    dir_work = r"/home/ubuntu/_working_dir"
    dir_out = r"/home/ubuntu/_working_dir"
    dir_dpool = r"/home/ubuntu/datapool"
    data_name = "sim_gnssr_4hz"
    dir_sim_l1 = os.path.join(dir_dpool, "internal", "datacube", data_name, "dataset")
    dir_wb = os.path.join(dir_dpool, "internal", "datacube", "wb_esa_cci", "dataset")

    # """ # ------------------------------------------------------------------------------------------------------------
    # second tile list
    roi = "roi_2"

    ftiles = ["NA1000M_E054N066T6", "NA1000M_E060N054T6", "NA1000M_E066N048T6", "NA1000M_E072N042T6",
              "NA1000M_E078N018T6", "NA1000M_E084N024T6", "NA1000M_E090N030T6", "NA1000M_E096N036T6",
              "NA1000M_E102N078T6", "SA1000M_E066N030T6", "SA1000M_E072N036T6", "SA1000M_E078N042T6",
              "SA1000M_E084N048T6", "SA1000M_E090N054T6", "AF1000M_E018N060T6", "AF1000M_E024N066T6",
              "EU1000M_E030N006T6", "EU1000M_E036N012T6", "EU1000M_E042N018T6", "EU1000M_E048N006T6",
              "EU1000M_E054N012T6", "EU1000M_E060N018T6", "EU1000M_E066N006T6", "EU1000M_E072N012T6",
              "AS1000M_E012N036T6", "AS1000M_E018N030T6", "AS1000M_E024N024T6", #"AS1000M_E018N054T6",  # repacing "AS1000M_E012N054T6"
              "AS1000M_E030N018T6", "AS1000M_E036N018T6", "AS1000M_E042N024T6", "AS1000M_E048N030T6",
              "AS1000M_E054N018T6", "AS1000M_E060N024T6", "AS1000M_E066N030T6", "OC1000M_E060N072T6",
              "OC1000M_E066N066T6", "OC1000M_E072N060T6", "OC1000M_E078N054T6", "OC1000M_E084N060T6"]

    """ # --------------
    ftile = ["NA1000M_E054N066", "NA1000M_E084N024",
             "SA1000M_E066N030", "EU1000M_E060N018",
             "AS1000M_E066N030", "OC1000M_E060N072",
             "AF1000M_E018N060", "OC1000M_E078N054"]
    """ # --------------
    """ # ------------------------------------------------------------------------------------------------------------
    roi = "roi_3"
    ftiles = ["NA1000M_E060N072T6", "NA1000M_E072N018T6", "NA1000M_E084N042T6", "NA1000M_E096N054T6",
              "NA1000M_E090N084T6", "SA1000M_E078N072T6", "EU1000M_E030N006T6", "EU1000M_E066N030T6",
              "AS1000M_E024N042T6", "AS1000M_E030N030T6", "AS1000M_E054N012T6", "AS1000M_E078N072T6",
              "AF1000M_E066N072T6", "AF1000M_E054N060T6", "AF1000M_E048N054T6", "AF1000M_E042N042T6"]
    """ # ------------------------------------------------------------------------------------------------------------
    
    """
    roi = "roi_global"
    dir_tiles =r"/home/ubuntu/datapool/internal/datacube/sim_gnssr_2hz/dataset"
    files =glob.glob(os.path.join(dir_tiles, '*/*.nc'))
    ftiles_all = [os.path.basename(f)[7:-3] for f in files]
    ftiles = random.choices(ftiles_all, k=100)
    """
    sat_num = 4
    tx_mask_list = ['GLONASS']
    # ---------------------------
    log_start_time = datetime.now()
    print(log_start_time, " GNSS-R coverage statistics calculation started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work,  log_start_time.strftime("%Y-%m-%d_%H%M%S")+
                            "_coverage_statistics_sim_"+str(sat_num)+"_sat_list_"+roi+"_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------
    logging.info(ftiles)
    logging.info("------------------------------------")

    file = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-09-17_gnss-r_coverage_maps/338_sat_full_list.txt"
    with open(file, 'r') as f:
        items = f.read()
    sat_full_list = [x.strip() for x in items.split(",")]
    #"""    #---------------------------------------------------------------------------------------------
    file = r"/home/ubuntu/_working_dir/_M5_sat_list_roi_2.txt"
    with open(file, 'r') as f:
        items = f.read()
    sat_list = [x.strip() for x in items.split(",")]
    mean_coverage, mean_min_count = plot_coverage_stat_wrapper(ftiles, sat_list, dir_sim_l1, dir_out, dir_wb, tx_mask_list=tx_mask_list)
    print(str(mean_coverage)+"%  " + str(mean_min_count))
    """    #---------------------------------------------------------------------------------------------
    iteration_num = 1
    for i in range(iteration_num):        
        find_optimal_constellation(ftiles, sat_full_list, dir_sim_l1, dir_wb, dir_out,
                                  log_start_time.strftime("%Y-%m-%d_%H%M%S"), sat_num, roi, tx_mask_list=tx_mask_list, mp_num=7)
    """    #---------------------------------------------------------------------------------------------

    logging.info("=======================================================")
    logging.info("Total processing time " + str(datetime.now() - log_start_time))
    print(datetime.now(), "GNSS-R coverage statistics calculation is finished!")

if __name__ == "__main__":
    main()


