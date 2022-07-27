import os
from netCDF4 import Dataset, date2num, num2date
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
import numpy as np
from pygnssr.sim_gnssr.SimDataCube import SimDataCube, get_l1_vars_template
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.common.utils.gdalport import read_tiff, write_tiff, call_gdal_util,  update_metadata
from pygnssr.sim_gnssr.utils.coverage_statistics import cal_coverage_stat
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

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _gen_composites(ftile, pname, sat_list, dir_sim_l1, dir_wb, dir_tile_composites, tx_mask_list=None,
                    date_int=None, nan_val=None, meta=None, colormap_path=None, overwrite=False):

    try:
        dir_out = dir_tile_composites
        tile_obj = Equi7Tile(ftile)
        sgrid = tile_obj.sgrid

        dst_geotags = tile_obj.get_tile_geotags()
        if nan_val is not None:
            dst_geotags['no_data_val'] = nan_val
        if meta is not None:
            dst_geotags['metadata'] = meta

        for st, et in zip(date_int[0], date_int[1]):
            fname = pname + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                    "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + "_" + ftile + ".tif"
            if os.path.exists(os.path.join(dir_out, fname)) and not overwrite:
                print("Output file exists! ", fname)
                continue

            arr_count_land, arr_count = cal_coverage_stat(ftile, sat_list, dir_sim_l1, dir_wb, tx_mask_list=tx_mask_list)

            _write_dst(arr_count, fname, dir_out, dst_geotags, colormap_path=colormap_path)

    except Exception as e:
        print(ftile + "  processing failed! .." + "\n" + str(e))

def _write_dst(arr, fname, dir_out, geotags,  colormap_path=None):
    """
    :param arr: numpy array
    :param fname:
    :param dir_out:
    :param geotags:
    :return:
    """

    if colormap_path is not None:
        ct = gen_gdal_ct(colormap_path)
    else:
        ct = None

    # metadata
    # -------------------------------------------
    os.makedirs(dir_out, exist_ok=True)
    dst_file = os.path.join(dir_out, fname)
    write_tiff(dst_file, np.rot90(arr,  k=1), tiff_tags=geotags, ct=ct)


def _merge_tiles(grid_res, sgrid_id, pname, dir_data, dir_out, gdal_path, nan_val,
                 meta=None, date_int=None, colormap_path=None, overwrite=False):

    res_dic = {"1000": "0.008999", "3000": "0.026997", "6000": "0.053995",
               "12000": "0.107991", "24000": "0.215982", "30000": "0.269978"}
    if str(grid_res) not in res_dic.keys():
        raise ValueError("The given grid spacing " + str(grid_res) + " is not supported!")
    else:
        grid_res_degree = res_dic[str(grid_res)]

    name_list = []
    for st, et in zip(date_int[0], date_int[1]):
        # check if start and end time are list (in case of climatologic has been set as True)
        if type(st) is list:
            name = pname.upper() + "_" + st[0].strftime("%9999%m%dT%H%M%S") + \
                   "_" + (et[0]-timedelta(microseconds=1)).strftime("%9999%m%dT%H%M%S") + \
                   "_" + sgrid_id + str(grid_res)+'M'
        else:
            name = pname.upper() + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                   "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + \
                   "_" + sgrid_id + str(grid_res)+'M'

        if os.path.exists(os.path.join(dir_out, name, ".tif")) and not overwrite:
            print("Output file exists! ", name)
            continue
        name_list.append(name)

    for name in name_list:
        pattern = name + "_*.tif"
        files_in = glob.glob(os.path.join(dir_data, pattern))

        file_merged = os.path.join(dir_out,  name + ".tif")
        os.makedirs(os.path.dirname(file_merged), exist_ok=True)
        p1 = subprocess.Popen(['python', os.path.join(gdal_path, 'gdal_merge.py'),
                               '-n', str(nan_val), '-co', 'COMPRESS=LZW', '-init', str(nan_val),
                               '-a_nodata', str(nan_val), '-o', file_merged] + files_in)

        p1.communicate()
        update_metadata(file_merged, meta, colormap_path=colormap_path)

        file_reproj = os.path.splitext(file_merged)[0]+'_EPSG4326.tif'
        # call gdalwarp for resampling
        p2 = subprocess.Popen([os.path.join(gdal_path, 'gdalwarp'),
                               '-of', 'GTiff', '-t_srs', 'EPSG:4326', '-r', 'near', '-co', 'COMPRESS=LZW',
                               '-tr', grid_res_degree, grid_res_degree,
                               '-te',  '-180', '-90', '180', '90', file_merged, file_reproj])

        p2.communicate()
        sleep(2)
        update_metadata(file_merged, meta, colormap_path=colormap_path)


def _merge_sgrids(grid_res, pname, sat_list, dir_data, dir_work, dir_out, gdal_path, nan_val,
                  meta=None, date_int=None, colormap_path=None, overwrite=False):

    sat_num = len(sat_list)
    name_list=[]
    for st, et in zip(date_int[0], date_int[1]):
        # check if start and end time are list (in case of climatologic has been set as True)
        if type(st) is list:

            name = pname.upper() + "_" + st[0].strftime("%9999%m%dT%H%M%S") + \
                   "_" + (et[0]-timedelta(microseconds=1)).strftime("%9999%m%dT%H%M%S")
        else:
            name = pname.upper() + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                   "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S")

        if os.path.exists(os.path.join(dir_out,  name + "_"+str(grid_res)+"M.tif")) and not overwrite:
            print("Output file exists! ", name)
            continue
        name_list.append(name)

    for name in name_list:
        file_merged = os.path.join(dir_work,  str(sat_num).zfill(3) + "sat_"+ name + "_"+str(grid_res)+"M.tif")
        file_dst = os.path.join(dir_out,  name + "_"+str(grid_res)+"M.tif")

        pattern = name + "_*"+str(grid_res)+"M_EPSG4326.tif"
        files_in = glob.glob(os.path.join(dir_data, pattern))
        os.makedirs(os.path.dirname(file_merged), exist_ok=True)
        p = subprocess.Popen(['python', os.path.join(gdal_path, 'gdal_merge.py'),
                              '-n', str(nan_val), '-co', 'COMPRESS=LZW', '-init', str(nan_val),
                              '-a_nodata', str(nan_val), '-o', file_merged] + files_in)
        p.communicate()
        sleep(2)
        update_metadata(file_merged, meta, colormap_path=colormap_path)
        sleep(2)
        os.makedirs(os.path.dirname(file_dst), exist_ok=True)
        shutil.move(file_merged, file_dst)
        sleep(2)




def coverage_plot_wrapper(grid_res, sat_list, dir_work, dir_dpool, dir_sim_l1, dir_wb, dir_out=None, out_grid_res=None,
                          tx_mask_list=None, stime=None, etime=None,  int_type='daily', overwrite=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, "SIM L2 coverage map production started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_simulated_coverage_mapper_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    etime = datetime.strptime(etime, "%Y-%m-%d %H:%M:%S")
    stime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S")

    if out_grid_res is None:
        out_grid_res = grid_res
    elif int(out_grid_res) not in [3000, 6000, 12000]:
        raise ValueError("Given output grid resolution is not supported!")

    pname = 'SIM_L2_NUMOBS'
    dir_merged = os.path.join(dir_work, pname.upper() + '_' + str(out_grid_res) + 'M_'+int_type.lower()+ '_tiles_merged')
    dir_global = os.path.join(dir_work, pname.upper() + '_' + str(out_grid_res) + 'M_'+int_type.lower()+'_global')
    gdal_path = r"/home/ubuntu/miniconda3/envs/en1/bin"
    if dir_out is None:
        dir_out = dir_work

    # ---------------------------------------------------------------------------------------------------------
    nan_val = 0
    scale_factor = 1
    data_version = 0.1
    if int_type is None:
        int_name = 'Total'
    else:
        int_name = int_type
    meta = {"product_name": int_name + " Number of simulated GNSS-R data",
            "scale_factor": str(scale_factor),
            "creator": "SPIRE GLOBAL",
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "Simulated GNSS-R observations",
            "varsion": str(data_version)}
    # colormap_path = os.path.join(dir_dpool, "internal", "misc", "color_tables", "gdal", "ct_cssm.ct")
    colormap_path = None
    # ---------------------------------------------------------------------------------------------------------

    try:
        date_int = get_time_intervals(stime, etime, interval_type=int_type)
        # output file names
        fnames = []
        for st, et in zip(date_int[0], date_int[1]):
            name = pname.upper() + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                   "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + "_" + str(out_grid_res)+"M.tif"
            if os.path.exists(os.path.join(dir_out, name)) and not overwrite:
                print("Output file exists! ", name)
                continue
            fnames.append(name)

        if ftile_list is None:
            sgrid_ids = ['AS', 'NA', 'SA', 'AF', 'EU', 'OC']
        else:
            sgrid_ids =list(set([ft[0:2] for ft in ftile_list]))

        for sgrid_id in sgrid_ids:
            if ftile_list is None:
                ftiles = get_ftile_names(dir_dpool, grid_res=out_grid_res, sgrid_ids=sgrid_id, land=True, eq=False)
            else:
                idx = np.char.find(ftile_list, sgrid_id) != -1
                ftiles = list(np.array(ftile_list)[idx])

            # calculate tile composites
            dir_tile_composites = os.path.join(dir_work, sgrid)

            # remove old folder
            #if os.path.exists(dir_tile_composites):
            #    shutil.rmtree(dir_tile_composites)

            #""" #-----------------------------------------------------------------------------------------------------
            if mp_num == 1:
                for ftile in ftiles:
                    _gen_composites(ftile, pname, sat_list, dir_sim_l1, dir_wb, dir_tile_composites,
                                    tx_mask_list=tx_mask_list, date_int=date_int,
                                    nan_val=nan_val, meta=meta, colormap_path=colormap_path, overwrite=overwrite)
            else:
                prod_ftile = partial(_gen_composites, pname=pname, sat_list=sat_list, dir_sim_l1=dir_sim_l1,
                                     dir_wb=dir_wb, dir_tile_composites=dir_tile_composites,
                                     tx_mask_list=tx_mask_list, date_int=date_int,
                                     nan_val=nan_val, meta=meta, colormap_path=colormap_path, overwrite=overwrite)
                p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)
            #""" #-----------------------------------------------------------------------------------------------------

            logging.info(sgrid + "  processed successfully! ..")

            _merge_tiles(out_grid_res, sgrid_id, pname, dir_tile_composites, dir_merged, gdal_path, nan_val,
                         meta=meta, date_int=date_int, colormap_path=colormap_path, overwrite=overwrite)
            #shutil.rmtree(dir_tile_composites)


        #_merge_sgrids(out_grid_res, pname, sat_list, dir_merged, dir_global, dir_out, gdal_path, nan_val,
        #             meta=meta, date_int=date_int, colormap_path=colormap_path, overwrite=overwrite)
        #shutil.rmtree(dir_merged)
        #shutil.rmtree(dir_global)

        logging.info("processed successfully! ..")
    except Exception as e:
        logging.error("processing failed! .." + "\n" + str(e))

    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    print(datetime.now(), "SIM coverage L2 data production finished!")


@click.command()
@click.argument('grid_res', default='3000', type=str)
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--out_grid_res', default=None, type=int,
              help='if set different than grid_res, it will be used for output grid resolution')
@click.option('--stime', default=None, type=str,
              help='Start date and time in following format: "%Y-%m-%d %H:%M:%S"  Default is None. '
                   'If provided, then the "days" argument will be overridden!')
@click.option('--etime', default=None, type=str,
              help='End date and time in following format: "%Y-%m-%d %H:%M:%S"  Default is None.'
                   'If not provided, then current date will be used as end time')
@click.option('--int_type', default='daily', type=str,
              help='Time interval that should be considered for making averages of measurements')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dir_work, dir_dpool, dir_out, out_grid_res, stime, etime, int_type, overwrite, mp_num):
    """
    This program make daily global mean of SSM maps in geotiff format from CYGNSS and sMAP observations

    """
    grid_res = 3000
    stime = datetime(2020, 9, 11).strftime("%Y-%m-%d %H:%M:%S")
    etime = datetime(2020, 9, 12).strftime("%Y-%m-%d %H:%M:%S")
    dir_out = dir_work
    int_type = 'daily'
    mp_num = 16

    dir_out = r"/home/ubuntu/_working_dir"
    file = r"/home/ubuntu/_working_dir/_M4_sat_list_roi_2.txt"

    dir_sim_l1 = os.path.join(dir_dpool, "internal", "datacube", "sim_gnssr_2hz", "dataset")
    dir_wb = os.path.join(dir_dpool, "internal", "datacube", "wb_esa_cci", "dataset")
    with open(file, 'r') as f:
        items = f.read()
    sat_list = [x.strip() for x in items.split(",")]
    tx_mask_list = ['GLONASS']

    coverage_plot_wrapper(grid_res, sat_list, dir_work, dir_dpool, dir_sim_l1, dir_wb, dir_out=dir_out, out_grid_res=out_grid_res,
                          tx_mask_list=tx_mask_list, stime=stime, etime=etime,  int_type=int_type, overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()


