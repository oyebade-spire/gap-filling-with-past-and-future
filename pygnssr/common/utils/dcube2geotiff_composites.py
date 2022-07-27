import os
from netCDF4 import Dataset, num2date, date2num, date2index
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
import numpy as np
from pygnssr.ssm_combined.CombSSMDataCube import CombSSMDataCube
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.common.utils.gdalport import read_tiff, write_tiff, gen_gdal_ct, call_gdal_util, update_metadata
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

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _gen_composites(ftile, dcube_name, var_name, stat_op,
                    dirs=None, dates_int=None, nan_val=None, meta=None, overwrite=False):

    try:
        dir_out = dirs['tile_composites']
        file_out_prefix = dcube_name.upper() +  '_' + var_name.upper() + '_' + stat_op.upper()

        fname_2proc = []
        for st, et in zip(dates_int[0], dates_int[1]):
            fname = file_out_prefix + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                    "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + "_" + ftile + ".tif"
            if os.path.exists(os.path.join(dir_out, fname)) and not overwrite:
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

        file_in = os.path.join(dirs['dcube'], sgrid, dcube_name.upper()+ '_' + ftile + '.nc')
        nc = Dataset(file_in, 'r')

        # read water bodies data to apply water mask
        # ------------------------------------------
        wb_file = os.path.join(dirs['wb'], sgrid, 'ESACCI-WB_' + ftile + '.tif')
        wb_arr, wb_tiff_tags = read_tiff(wb_file)
        # rotate the water bodies raster image
        wb_arr = np.rot90(wb_arr, k=-1)

        # read data fields
        # todo generalize it for other variables and datasets
        # TEMP ------------------------------------------------------------------------

        #var = nc.variables[var_name]
        #var = nc.variables['wet'][:, :] - nc.variables['dry'][:, :]
        #dst_arr = np.full_like(var[:, :], nan_val)
       # mask = np.ma.getmaskarray(nc.variables['dry'])[0]
        #dst_arr[~mask] = var[~mask]
        #_write_dst(dst_arr, fname, dir_out, dst_geotags, wb_arr, masking_water=False, colormap_path=None)


        # TEMP ------------------------------------------------------------------------
        t = nc.variables['ddm_timestamp_utc'][:, :, :]
        units = nc.variables['ddm_timestamp_utc'].units
        calendar = nc.variables['ddm_timestamp_utc'].calendar
        #snr = nc.variables['ddm_snr'][:, :, :]
        rfl = nc.variables['rfl'][:, :, :]


        """
        #------------------------------------------------------------------------
        t = nc.variables['sample_time'][:, :, :]
        units = nc.variables['sample_time'].units
        calendar = nc.variables['sample_time'].calendar
        snr = nc.variables['reflect_snr_at_sp'][:, :, :]
        rfl = 10.0 * np.log10(nc.variables['reflectivity_at_sp'][:, :, :])
       
        qflags = nc.variables['quality_flags'][:, :, :]
        mask = qflags != 0
        ant_corr_invalid = np.bitwise_and(qflags, 128) != 0
        snr_mask = np.bitwise_and(qflags, 256) != 0
        rfi_mask = np.bitwise_and(qflags, 512) != 0
        invalid = snr_mask | rfi_mask

        filter = rfi_mask
        snr = np.ma.masked_where(filter, snr)
        rfl = np.ma.masked_where(filter, rfl)
        #------------------------------------------------------------------------
        """
        # todo: generalize these as well
        if var_name.lower() == 'snr':
            var = snr
        elif var_name.lower() == 'rfl':
            var = rfl
        elif var_name.lower() == 'obs':
            var = snr
        else:
            raise   ValueError(var_name + ' is not supported!')

        for fname in  fname_2proc:
            st = datetime.strptime(fname.split('_')[4], "%Y%m%dT%H%M%S")
            et = datetime.strptime(fname.split('_')[5], "%Y%m%dT%H%M%S")

            # initialize destination array
            dst_arr = np.full_like(var[0, :, :], nan_val)
            if len(var) > 0:
                var_stat = _cal_var_stat(st, et, var, t, units, calendar, stat_op)
                dst_arr[~var_stat.mask] = var_stat[~var_stat.mask]

            _write_dst(dst_arr, fname, dir_out, dst_geotags, wb_arr, masking_water=False, colormap_path=None)
            

        return True
    except Exception as e:
        logging.error(ftile + "\n" + str(e))
        return False

def _cal_var_stat(start_time, end_time, var, nc_time, unit, calendar, stat_op):
    st_num = date2num(start_time, units=unit, calendar=calendar)
    et_num = date2num(end_time, units=unit, calendar=calendar)
    val = (nc_time[:, :, :] >= st_num)&(nc_time[:, :, :] < et_num)
    var_out = np.ma.masked_where(~val, var)
    return_var = getattr(var_out, stat_op)(axis=0)
    if stat_op == 'count':
        return_var = np.ma.masked_where(return_var == 0, return_var)
    return return_var


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


def merge_tile_composites(grid_res, fname_prefix, sgrid_id, dir_data, dir_out, gdal_path, nan_val,
                          meta=None, dates_int=None, colormap_path=None, overwrite=False, epsg4326=False):

    res_dic = {"1000": "0.008999", "3000": "0.026997", "6000": "0.053995",
               "12000": "0.107991", "24000": "0.215982", "30000": "0.269978"}
    if str(grid_res) not in res_dic.keys():
        raise ValueError("The given grid spacing " + str(grid_res) + " is not supported!")
    else:
        grid_res_degree = res_dic[str(grid_res)]

    name_list = []
    for st, et in zip(dates_int[0], dates_int[1]):
        # check if start and end time are list (in case of climatologic has been set as True)
        if type(st) is list:
            name = fname_prefix + "_" + st[0].strftime("%9999%m%dT%H%M%S") + \
                   "_" + (et[0]-timedelta(microseconds=1)).strftime("%9999%m%dT%H%M%S") + \
                   "_" + sgrid_id + str(grid_res)+'M'
        else:
            name = fname_prefix + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                   "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + \
                   "_" + sgrid_id + str(grid_res)+'M'

        if os.path.exists(os.path.join(dir_out, name + '_EPSG4326.tif')) and not overwrite:
            print("Output file exists! ", name)
            continue
        else:
            name_list.append(name)

    for name in name_list:
        try:
            pattern = name + "_*.tif"
            files_in = glob.glob(os.path.join(dir_data, pattern))
            file_merged = os.path.join(dir_out,  name + ".tif")

            options = {'-n':str(nan_val),
                       '-co':'COMPRESS=LZW',
                       '-init':str(nan_val),
                       '-a_nodata':str(nan_val),
                       '-o':file_merged}
            call_gdal_util('gdal_merge', gdal_path=gdal_path, src_files=files_in, options=options)
            update_metadata(file_merged, meta, colormap_path=colormap_path)

            if epsg4326:
                file_reproj = os.path.join(dir_out,  name + '_EPSG4326.tif')
                # call gdalwarp for resampling
                options = {'-of': 'GTiff',
                           '-t_srs': 'EPSG:4326',
                           '-r': 'near',
                           '-co': 'COMPRESS=LZW',
                           '-tr': grid_res_degree + " "+ grid_res_degree,
                           '-overwrite': " ",
                           '-te': "-180 -90 180 90"}
                call_gdal_util('gdalwarp', gdal_path=gdal_path, src_files=file_merged,
                               dst_file=file_reproj, options=options)
                os.remove(file_merged)

        except Exception as e:
            logging.error(name + "\n" + str(e))


def merge_sgrid_composites(grid_res, fname_prefix, dir_data, dir_work, dir_out, gdal_path, nan_val,
                           meta=None, dates_int=None, colormap_path=None, overwrite=False):

    name_list=[]
    for st, et in zip(dates_int[0], dates_int[1]):
        # check if start and end time are list (in case of climatologic has been set as True)
        if type(st) is list:
            name = fname_prefix + "_" + st[0].strftime("%9999%m%dT%H%M%S") + \
                   "_" + (et[0]-timedelta(microseconds=1)).strftime("%9999%m%dT%H%M%S")
        else:
            name = fname_prefix + "_" + st.strftime("%Y%m%dT%H%M%S") + \
                   "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S")
        # todo this is already done in wrapper code!
        if os.path.exists(os.path.join(dir_out,  name[14:20], name + "_"+str(grid_res)+"M.tif")) and not overwrite:
            print("Output file exists! ", name)
            continue
        name_list.append(name)

    for name in name_list:
        try:
            file_merged = os.path.join(dir_work,  name + "_"+str(grid_res)+"M.tif")
            file_dst = os.path.join(dir_out,  name[14:20], name + "_"+str(grid_res)+"M.tif")

            pattern = name + "_*"+str(grid_res)+"M_*.tif"
            files_in = glob.glob(os.path.join(dir_data, pattern))
            os.makedirs(os.path.dirname(file_merged),


                        exist_ok=True)

            options = {'-n':str(nan_val),
                       '-co':'COMPRESS=LZW',
                       '-init':str(nan_val),
                       '-a_nodata':str(nan_val),
                       '-o':file_merged}

            call_gdal_util('gdal_merge', gdal_path=gdal_path, src_files=files_in, options=options)
            update_metadata(file_merged, meta, colormap_path=colormap_path)

            os.makedirs(os.path.dirname(file_dst), exist_ok=True)
            shutil.move(file_merged, file_dst)

        except Exception as e:
            logging.error(name + "\n" + str(e))


def _wrapper(dcube_name, var_name, dir_work, dir_dpool, dir_out, ftile_list=None, grid_res=None,
             stime=None, etime=None, int_type=None, comp_op=None, overwrite=False, water_mask=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, "Data composite production started from python code ...")

    etime = datetime.strptime(etime, "%Y-%m-%d %H:%M:%S")
    stime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S")

    stat_operations = comp_op if isinstance(comp_op, list) else [comp_op]

    for stat_op in stat_operations:
        dirs = _get_dirs(dcube_name, dir_work=dir_work, dir_dpool=dir_dpool, dir_out=dir_out)
        fname_prefix = dcube_name.upper() +  '_' + var_name.upper() + '_' + stat_op.upper()
        dirs['tiles_merged'] = os.path.join(dirs['work'], fname_prefix + '_' + str(grid_res) + 'M' + '_tiles_merged')
        dirs['sgrids_merged'] = os.path.join(dirs['work'], fname_prefix + '_' + str(grid_res) + 'M' + '_global')
        # todo: handle color table
        dirs['colormap_cssm'] = os.path.join(dirs['dpool'], "internal", "misc", "color_tables", "gdal", "ct_cssm.ct")
        dirs['tiles_list'] = os.path.join(dirs['dpool'], "internal", "misc", "land_tile_list")

        # setup logging ----------------------------------------------------------------------------------------
        log_file = os.path.join(dirs['work'], datetime.now().strftime("%Y-%m-%d_%H%M%S") +
                                "_" +dcube_name + "_data_composite_production_log_file.log")
        log_level = logging.INFO
        log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
        log_datefrmt = "%Y-%m-%d %H:%M:%S"
        logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
        # setup logging ----------------------------------------------------------------------------------------

        if grid_res is None:
            grid_res = 3000
        elif int(grid_res) not in [3000, 6000]:
            raise ValueError("Given output grid resolution is not supported!")

        # ---------------------------------------------------------------------------------------------------------
        nan_val = -9999.0
        scale_factor = 1
        data_version = 0.1
        if int_type is None:
            int_name = 'Total'
        else:
            int_name = int_type
        # todo: edit meta data
        meta = {"product_name": int_name + " " + stat_op.upper() + " of " + var_name,
                "scale_factor": str(scale_factor),
                "units": "",
                "creator": "SPIRE GLOBAL",
                "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "",
                "varsion": str(data_version)}
        # ---------------------------------------------------------------------------------------------------------
        try:
            dates_int = get_time_intervals(stime, etime, interval_type=int_type)
            if not overwrite:
                # output file names
                dates_int_2proc = cpy(dates_int)
                for st, et in zip(dates_int[0], dates_int[1]):

                    name = dcube_name.upper() +  '_' + var_name.upper() + '_' + stat_op.upper() + \
                           "_" + st.strftime("%Y%m%dT%H%M%S") + \
                           "_" + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S") + "_" + str(grid_res)+"M.tif"
                    if os.path.exists(os.path.join(dirs['out'], name)) and not overwrite:
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
                    sgrid = sgrid_id.upper()+str(grid_res)+"M"
                    if ftile_list is None:
                        ftiles = get_ftile_names(dir_dpool, grid_res=grid_res, sgrid_ids=sgrid_id, land=True, eq=True)
                    else:
                        idx = np.char.find(ftile_list, sgrid_id) != -1
                        ftiles = list(np.array(ftile_list)[idx])

                    # calculate tile composites
                    dirs['tile_composites'] = os.path.join(dirs['work'], sgrid)
                    if mp_num == 1:
                        for ftile in ftiles:
                            succed = _gen_composites(ftile, dcube_name, var_name, stat_op, dirs=dirs,
                                            dates_int=dates_int_2proc, nan_val=nan_val, meta=meta,  overwrite=overwrite)
                    else:
                        prod_ftile = partial(_gen_composites, dcube_name=dcube_name, var_name=var_name, stat_op=stat_op,
                                             dirs=dirs, dates_int=dates_int_2proc, nan_val=nan_val,
                                             meta=meta,  overwrite=overwrite)
                        p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)
                    logging.info(sgrid + "  tile composites generation is finished! ...")

                for sgrid_id in ['AS', 'NA', 'SA', 'AF', 'EU', 'OC']:
                    sgrid = sgrid_id.upper()+str(grid_res)+"M"
                    dirs['tile_composites'] = os.path.join(dirs['work'], sgrid)
                    merge_tile_composites(grid_res, fname_prefix, sgrid_id, dirs['tile_composites'],
                                          dirs['tiles_merged'], dirs['gdal_path'],
                                          nan_val, meta=meta, dates_int=dates_int_2proc,
                                          colormap_path=dirs['colormap_cssm'], overwrite=overwrite)
                    logging.info(sgrid + "  merging of tiles to sub-grid is finished! .")

                merge_sgrid_composites(grid_res, fname_prefix, dirs['tiles_merged'], dirs['sgrids_merged'],
                                       dirs['out'], dirs['gdal_path'], nan_val, meta=meta,
                                       dates_int=dates_int_2proc, colormap_path=dirs['colormap_cssm'], overwrite=overwrite)
                logging.info("Merging of sub-grids is finished! ...")
                #shutil.rmtree(dir_merged)
                #shutil.rmtree(dir_global)

        except Exception as e:
                logging.error("processing failed! .." + "\n" + str(e))

    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    print(datetime.now(), "L3U1 data production finished!")


def _get_dirs(dcube_name, dir_work=None, dir_dpool=None, dir_out=None):
    dirs={}
    dirs['home'] = r"/home/ubuntu/"
    dirs['gdal_path'] = os.path.join(dirs['home'], 'miniconda3', 'envs', 'en1', 'bin')
    dirs['work'] = os.path.join(dirs['home'], '_working_dir') if dir_work is None else dir_work
    dirs['dpool'] = os.path.join(dirs['home'], 'datapool') if dir_dpool is None else dir_dpool
    dirs['lc'] = os.path.join(dirs['dpool'], 'external', 'landcover_esa_cci')
    dirs['wb'] = os.path.join(dirs['dpool'], "internal", "datacube", "wb_esa_cci", "dataset")

    if dir_out is None:
        dirs['out'] = os.path.join(dir_dpool, "internal", "datacube", dcube_name, "composites")
    else:
        dirs['out'] = dir_out

    if dcube_name == 'cygnss_l1':
        dirs['dcube'] = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L1")
    elif dcube_name == 'cygnss_l2':
        dirs['dcube'] = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2")
    elif dcube_name == 'cygnss_l2p':
        dirs['dcube'] = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2P")
    elif dcube_name == 'spire_gnssr_l1':
        dirs['dcube'] = os.path.join(dir_dpool, "internal", "datacube", "spire_gnssr", "prod-0.3.7", "dataset", "L1")
    elif dcube_name == 'comb_ssm_l2u1':
        dirs['dcube'] = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U1")

    return dirs


@click.command()
@click.option('--dcube_name', default="cygnss_l1", type=str,
              help='DataCube name (e.g. "cygnss_l1", "spire_gnssr_l1"')
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--grid_res', default=3000, type=int,
              help='input and output grid resolution')
@click.option('--stime', default=None, type=str,
              help='Start date and time in following format: "%Y-%m-%d %H:%M:%S"')
@click.option('--etime', default=None, type=str,
              help='End date and time in following format: "%Y-%m-%d %H:%M:%S"'
                   'If not provided, then current date will be used as end time')
@click.option('--int_type', default=None, type=str,
              help='Time intervals that should be considered for runing the calcualtions.'
                   'Possible intrval types: "daily", "monthly", "decadal", "seasonally".'
                   'Default is None, returning a single interval starting with stime and ending with etime')
@click.option('--comp_op', default='mean', type=str,
              help='A list of compositing operation. Supported operation: ["mean", "stdev"].')
@click.option('--var_name', default='snr', type=str,
              help='Variable name. Supported variables: "snr", "rfl", "obs".')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--water_mask', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(dcube_name, dir_work, dir_dpool, dir_out, grid_res, stime, etime, int_type, comp_op, var_name,
         overwrite, water_mask, mp_num):
    """
    This program make daily global mean of SSM maps in geotiff format from CYGNSS and sMAP observations

    """
    ftile_list = None
    dcube_name = 'cygnss_l2'
    mp_num = 7
    stime = datetime(2018, 1, 1).strftime("%Y-%m-%d %H:%M:%S")
    etime = datetime(2021, 1, 1).strftime("%Y-%m-%d %H:%M:%S")
    comp_op = 'mean'
    var_name = 'rfl'
    _wrapper(dcube_name, var_name, dir_work, dir_dpool, dir_out, ftile_list=ftile_list, grid_res=grid_res,
             stime=stime, etime=etime, int_type=int_type, comp_op=comp_op,
             overwrite=overwrite, water_mask=water_mask, mp_num=mp_num)


if __name__ == "__main__":
    main()


