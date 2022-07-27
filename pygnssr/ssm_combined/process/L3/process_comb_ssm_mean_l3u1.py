import os
from netCDF4 import num2date, date2num, date2index
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
import numpy as np
from pygnssr.ssm_combined.CombSSMDataCube import CombSSMDataCube
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.common.utils.gdalport import read_tiff, write_tiff, call_gdal_util, update_metadata, gen_gdal_ct
from pygnssr.common.utils.dcube2geotiff_composites import merge_sgrid_composites, merge_tile_composites
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
from copy import deepcopy as cpy
from pygnssr.common.utils.e7_geotiff_merge import e7_geotiff_merge

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _gen_composites(ftile, dir_comb_ssm_l2u1, dir_wb, dir_tile_composites,
                    dates_int=None, nan_val=None, meta=None, colormap_path=None, overwrite=False):

    try:
        fname_2proc = []
        for st, et in zip(dates_int[0], dates_int[1]):
            fname = "COMB-SSM_L3U1_" + st.strftime("%Y%m%dT%H%M%S") + \
                    "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + "_" + ftile + ".tif"
            if os.path.exists(os.path.join(dir_tile_composites, fname)) and not overwrite:
                print("Output file exists! ", fname)
                continue
            else:
                fname_2proc.append(fname)

        if len(fname_2proc) == 0:
            return True, 1

        tile_obj = Equi7Tile(ftile)
        sgrid = tile_obj.sgrid

        dst_geotags = tile_obj.get_tile_geotags()
        if nan_val is not None:
            dst_geotags['no_data_val'] = nan_val
        if meta is not None:
            dst_geotags['metadata'] = meta

        dc = CombSSMDataCube(ftile, "L2U1", os.path.join(dir_comb_ssm_l2u1, sgrid), flag='r')

        # read water bodies data to apply water mask
        # ------------------------------------------
        wb_file = os.path.join(dir_wb, sgrid, 'ESACCI-WB_' + ftile + '.tif')
        wb_arr, wb_tiff_tags = read_tiff(wb_file)
        # rotate the water bodies raster image
        wb_arr = np.rot90(wb_arr, k=-1)

        # define weights for averaging of sm products
        w_smap = 1
        w_cygnss = 0

        # read and encode smap soil moisture data
        sm_smap = dc.nc['smap']['sm'][:, :, :]
        t_smap = dc.nc['smap']['time_utc'][:, :, :]
        un_smap = dc.nc['smap']['time_utc'].units
        cl_smap = dc.nc['smap']['time_utc'].calendar

        # read and encode cygnss soil moisture data
        sm_cygnss = dc.nc['cygnss']['cssm'][:, :, :]
        t_cygnss = dc.nc['cygnss']['time_utc'][:, :, :]
        un_cygnss = dc.nc['cygnss']['time_utc'].units
        cl_cygnss = dc.nc['cygnss']['time_utc'].calendar

        for fname in  fname_2proc:
            st = datetime.strptime(fname.split('_')[2], "%Y%m%dT%H%M%S")
            et = datetime.strptime(fname.split('_')[3], "%Y%m%dT%H%M%S")

            # initialize destination array
            dst_arr = np.full_like(sm_smap[0, :, :], nan_val)
            if len(sm_smap) > 0:
                sm_mean_smap = _cal_sm_mean(st, et, sm_smap, t_smap, un_smap, cl_smap)
                dst_arr[~sm_mean_smap.mask] = sm_mean_smap[~sm_mean_smap.mask]

            if len(sm_cygnss) > 0:
                sm_mean_cygn = _cal_sm_mean(st, et, sm_cygnss, t_cygnss, un_cygnss, cl_cygnss)
                dst_arr[~sm_mean_cygn.mask] = sm_mean_cygn[~sm_mean_cygn.mask]

            if len(sm_smap) > 0 and len(sm_cygnss) > 0:
                merge_idx = ~sm_mean_smap.mask & ~sm_mean_cygn.mask
                dst_arr[merge_idx] = (sm_mean_smap[merge_idx]*w_smap + sm_mean_cygn[merge_idx]*w_cygnss)/(w_smap+w_cygnss)

            nan_ind = np.where(dst_arr == nan_val)
            # encode results using the scale factor
            dst_arr = (dst_arr * float(meta['scale_factor'])).round().astype('uint8')
            # reassign NaN values after encoding
            dst_arr[nan_ind] = nan_val

            _write_dst(dst_arr, fname, dir_tile_composites, dst_geotags, wb_arr, masking_water=True, colormap_path=colormap_path)
        return True

    except Exception as e:
        logging.error(ftile + "\n" + str(e))
        return False


def _cal_sm_mean(start_time, end_time, sm, nc_time, unit, calendar):
    st_num = date2num(start_time, units=unit, calendar=calendar)
    et_num = date2num(end_time, units=unit, calendar=calendar)
    #val = np.ma.masked_greater_equal(np.ma.masked_less(nc_time[:, :, :], st_num), et_num)
    #var = np.ma.masked_where(np.ma.getmask(val), sm)
    val = (nc_time[:, :, :] >= st_num)&(nc_time[:, :, :] < et_num)
    var = np.ma.masked_where(~val, sm)
    return var.mean(axis=0)


def _write_dst(arr, fname, dir_out, geotags,  wb_arr, masking_water=False, colormap_path=None):
    """
    :param arr: numpy array
    :param fname:
    :param dir_out:
    :param geotags:
    :param wb_arr:
    :param masking_water:
    :return:
    """

    # Masking
    if masking_water:
        idx = (wb_arr == 2)
        arr[idx] = geotags['no_data_val']

    if colormap_path is not None:
        ct = gen_gdal_ct(colormap_path)
    else:
        ct = None

    # metadata
    # -------------------------------------------
    os.makedirs(dir_out, exist_ok=True)
    dst_file = os.path.join(dir_out, fname)
    write_tiff(dst_file, np.rot90(arr,  k=1), tiff_tags=geotags, ct=ct)


def comb_ssm_l3u1_wrapper(dir_work, dir_dpool, dir_out=None, ftile_list=None, out_grid_res=None,
                          stime=None, etime=None, days=14, int_type='daily', overwrite=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, "COMB-SSM L3U1 data production started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_COMB-SSM_L3U1_data_production_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    if etime is None:
        etime = datetime.now()
    else:
        etime = datetime.strptime(etime, "%Y-%m-%d %H:%M:%S")
    if stime is None:
        stime = etime - timedelta(days)
    else:
        stime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S")

    if out_grid_res is None:
        out_grid_res = 6000
    elif int(out_grid_res) not in [3000, 6000]:
        raise ValueError("Given output grid resolution is not supported!")

    fname_prefix = 'COMB-SSM_L3U1'
    dir_comb_ssm_l2u1 = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U1")
    dir_wb = os.path.join(dir_dpool, "internal", "datacube", "wb_esa_cci", "dataset")
    dir_merged = os.path.join(dir_work, fname_prefix+ '_' + str(out_grid_res) + 'M_'+int_type.lower()+ '_tiles_merged')
    dir_global = os.path.join(dir_work, fname_prefix + '_' + str(out_grid_res) + 'M_'+int_type.lower()+'_global')
    gdal_path = r"/home/ubuntu/miniconda3/envs/en1/bin"
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L3U1")

    # ---------------------------------------------------------------------------------------------------------
    nan_val = 255
    scale_factor = 100
    data_version = 0.3
    if int_type is None:
        int_name = 'Total'
    else:
        int_name = int_type
    meta = {"product_name": int_name + " Mean of SMAP+CYGNSS Soil Moisture Products",
            "scale_factor": str(scale_factor),
            "units": "cm³/cm³",
            "creator": "SPIRE GLOBAL",
            "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "SMAP Level-3 SM 9km, CYGNSS level-1 Reflectivity",
            "varsion": str(data_version)}
    colormap_path = os.path.join(dir_dpool, "internal", "misc", "color_tables", "gdal", "ct_cssm.ct")
    # ---------------------------------------------------------------------------------------------------------
    try:
        dates_int = get_time_intervals(stime, etime, interval_type=int_type)
        if not overwrite:
            # output file names
            dates_int_2proc = cpy(dates_int)
            for st, et in zip(dates_int[0], dates_int[1]):
                name = fname_prefix + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                       "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + "_" + str(out_grid_res)+"M.tif"
                if os.path.exists(os.path.join(dir_out, name[14:20], name)) and not overwrite:
                    print("Output file exists! ", name)
                    dates_int_2proc[0].remove(st)
                    dates_int_2proc[1].remove(et)
        else:
            dates_int_2proc = dates_int

        if len(dates_int_2proc[0]) == 0:
            logging.info("All files for given dates are already available in output directory! "
                                 "Use overwrite keyword to reproduce them")
        else:
            if ftile_list is None:
                sgrid_ids = ['AS', 'NA', 'SA', 'AF', 'EU', 'OC']
            else:
                sgrid_ids =list(set([ft[0:2] for ft in ftile_list]))

            for sgrid_id in sgrid_ids:
                sgrid = sgrid_id.upper()+str(out_grid_res)+"M"
                if ftile_list is None:
                    ftiles = get_ftile_names(dir_dpool, grid_res=out_grid_res, sgrid_ids=sgrid_id, land=True, eq=False)
                else:
                    idx = np.char.find(ftile_list, sgrid_id) != -1
                    ftiles = list(np.array(ftile_list)[idx])

                dir_tile_composites = os.path.join(dir_work, sgrid)
                if mp_num == 1:
                    for ftile in ftiles:
                        succed = _gen_composites(ftile, dir_comb_ssm_l2u1, dir_wb, dir_tile_composites, dates_int=dates_int_2proc,
                                        nan_val=nan_val, meta=meta, colormap_path=colormap_path, overwrite=overwrite)
                else:
                    prod_ftile = partial(_gen_composites, dir_comb_ssm_l2u1=dir_comb_ssm_l2u1, dir_wb=dir_wb,
                                         dir_tile_composites=dir_tile_composites, dates_int=dates_int_2proc, nan_val=nan_val,
                                         meta=meta, colormap_path=colormap_path, overwrite=overwrite)
                    p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)
                logging.info(sgrid + "  tile composites generation is finished! ...")

        for sgrid_id in ['AS', 'NA', 'SA', 'AF', 'EU', 'OC']:
            sgrid = sgrid_id.upper()+str(out_grid_res)+"M"
            dir_tile_composites = os.path.join(dir_work, sgrid)
            merge_tile_composites(out_grid_res, fname_prefix, sgrid_id, dir_tile_composites, dir_merged, gdal_path,
                                  nan_val, meta=meta, dates_int=dates_int_2proc,
                                  colormap_path=colormap_path, overwrite=overwrite, epsg4326=True)
            logging.info(sgrid + "  merging of tiles to sub-grid is finished! .")

        merge_sgrid_composites(out_grid_res, fname_prefix, dir_merged, dir_global, dir_out, gdal_path, nan_val,
                               meta=meta, dates_int=dates_int_2proc,
                               colormap_path=colormap_path, overwrite=False)
        logging.info("Merging of sub-grids is finished! ...")
        shutil.rmtree(dir_merged)
        shutil.rmtree(dir_global)

    except Exception as e:
        logging.error("processing failed! .." + "\n" + str(e))

    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    logging.shutdown()
    # check log file to see if error has been logged
    with open(log_file, "r") as f:
        content = f.readlines()
    idx = np.char.find(content, "[ERROR]") != -1
    if idx.any():
        print('Error detected!')
        print(np.array(content)[idx])
        # todo: send an email alert
    else:
        print('No Error detected!')
        shutil.rmtree(dir_tile_composites)
    print(datetime.now(), "L3U1 data production finished!")


@click.command()
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--out_grid_res', default=6000, type=int,
              help='if set different than grid_res, it will be used for output grid resolution')
@click.option('--days', default=14, type=int,
              help='Number of days in the past to be used as time filter for searching '
                   'and downloading the data files. Default is 14 days')
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
def main(dir_work, dir_dpool, dir_out, out_grid_res, stime, etime, days, int_type, overwrite, mp_num):
    """
    This program make daily global mean of SSM maps in geotiff format from CYGNSS and sMAP observations

    """
    ftile_list = None
    comb_ssm_l3u1_wrapper(dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list, out_grid_res=out_grid_res,
                          stime=stime, etime=etime, days=days, int_type=int_type, overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()
