import os
from netCDF4 import num2date, date2num
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
from pygnssr.common.utils.gdalport import read_tiff
import numpy as np
from pygnssr.ssm_combined.CombSSMDataCube import CombSSMDataCube
from pygnssr.cygnss.CygnssDataCube import CygnssDataCube
from pygnssr.smap.SmapDataCube import SmapDataCube
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from datetime import datetime
import click
import logging
from functools import partial
import multiprocessing as mp
import json
from pygnssr.cygnss.cygnss_utils import get_mask


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _convert_nctime(nctime_src, units_dst, calendar_dst):

    mask = nctime_src[:, :, :].mask
    tt = np.ma.empty(nctime_src.shape, dtype=np.float64)
    tt.mask = mask
    dates = num2date(nctime_src[:, :, :], nctime_src.units, calendar=nctime_src.calendar)
    tt[~mask] = date2num(dates[~mask], units_dst, calendar=calendar_dst)
    return tt


def _copy_cygnss(dc, dc_cygn, ref_unit, ref_calendar, wb_mask):
    try:
        dc_cygn.read()
        dc.open_nc()
        water_mask = np.repeat(wb_mask[np.newaxis], dc_cygn.nc.dimensions['sample'].size, axis=0)
        for var_name in dc.nc['cygnss'].variables.keys():
            if var_name == 'time_utc':
                if dc_cygn.nc['ddm_timestamp_utc'].units == ref_unit and \
                        dc_cygn.nc['ddm_timestamp_utc'].calendar == ref_calendar:
                    dc.nc['cygnss']['time_utc'][:, :, :] = dc_cygn.nc['ddm_timestamp_utc']
                else:
                    dc.nc['cygnss']['time_utc'].units = ref_unit
                    dc.nc['cygnss']['time_utc'].calendar = ref_calendar
                    # convert nctime
                    dc.nc['cygnss']['time_utc'][:, :, :] = _convert_nctime(dc_cygn.nc['ddm_timestamp_utc'],
                                                                           ref_unit, ref_calendar)[:, :, :]
            elif var_name == 'rssm':
                #todo this part should have already done during L2 processing chain
                # apply mask
                qflags = dc_cygn.nc.variables['quality_flags'][:, :, :]
                rssm = dc_cygn.nc.variables['rssm']
                rssm = np.ma.masked_outside(rssm, 0, 100)
                cygnss_mask = get_mask(qflags)
                rssm = np.ma.masked_where(cygnss_mask, rssm)
                rssm = np.ma.masked_where(water_mask, rssm)
                dc.nc['cygnss'][var_name][:, :, :] = rssm[:, :, :]
            elif var_name == 'cssm':
                cssm = dc_cygn.nc.variables['cssm']
                cssm = np.ma.masked_where(water_mask, cssm)
                dc.nc['cygnss'][var_name][:, :, :] = cssm[:, :, :]
            elif var_name == 'sp_lon':
                sp_lon = (dc_cygn.nc.variables['sp_lon'][:, :, :]).astype('float64')
                # convert longitude in [0, 360] format to [-180°,180°]
                sp_lon = ((sp_lon - 180.0) % 360.0) - 180.0
                dc.nc['cygnss'][var_name][:, :, :] = sp_lon[:, :, :]
            else:
                dc.nc['cygnss'][var_name][:, :, :] = dc_cygn.nc[var_name][:, :, :]

        dc.close_nc()
        dc_cygn.close_nc()
        return True, ""
    except Exception as e:
        error_message = "Copying of CYGNSS tile failed ! .." + "\n" + str(e)
        return False, error_message


def _copy_smap(dc, dc_smap, ref_unit, ref_calendar, wb_mask):
    try:
        dc_smap.read()
        dc.open_nc()
        water_mask = np.repeat(wb_mask[np.newaxis], dc_smap.nc.dimensions['sample'].size, axis=0)
        for var_name in dc.nc['smap'].variables.keys():
            if var_name == 'time_utc':
                if dc_smap.nc['tb_time_utc'].units == ref_unit and dc_smap.nc['tb_time_utc'].calendar == ref_calendar:
                    dc.nc['smap']['time_utc'][:, :, :] = dc_smap.nc['tb_time_utc'][:, :, :]
                else:
                    dc.nc['smap']['time_utc'].units = ref_unit
                    dc.nc['smap']['time_utc'].calendar = ref_calendar
                    # convert nctime
                    dc.nc['smap']['time_utc'][:, :, :] = _convert_nctime(dc_smap.nc['tb_time_utc'],
                                                                         ref_unit, ref_calendar)[:, :, :]
            elif var_name == 'sm':
                sm = dc_smap.nc['soil_moisture'][:, :, :]
                sm = np.ma.masked_where(water_mask, sm)
                dc.nc['smap']['sm'][:, :, :] = sm
            else:
                dc.nc['smap'][var_name][:, :, :] = dc_smap.nc[var_name][:, :, :]
        dc.close_nc()
        dc_smap.close_nc()
        return True, ""
    except Exception as e:
        error_message = "Copying of SMAP tile failed ! .." + "\n" + str(e)
        return False, error_message



def _gen_comb_ssm_l2u1(ftile, dir_work, dir_cygnss_l2, dir_smap_l3, dir_wb, dir_out, overwrite=False):

    try:
        sgrid = ftile.split('_')[0]
        dc_dst = CombSSMDataCube(ftile, "L2U1", os.path.join(dir_out, sgrid))
        if os.path.exists(dc_dst.fpath) and not overwrite:
            # raise ValueError("Output file exists! Set overwrite keyword as True..." + dc_dst.fpath)
            print("Output file exists! Set overwrite keyword as True..." + dc_dst.fpath)
            return None

        dc = CombSSMDataCube(ftile, "L2U1", dir_work, flag='w', overwrite=True)
        dc.close_nc()
        dc_cygn = CygnssDataCube(ftile, "L2", os.path.join(dir_cygnss_l2, sgrid))
        dc_smap = SmapDataCube(ftile, "L3", os.path.join(dir_smap_l3, sgrid))

        # read water bodies data to apply water mask
        wb_file = os.path.join(dir_wb, sgrid, 'ESACCI-WB_' + ftile + '.tif')
        wb_arr, wb_tiff_tags = read_tiff(wb_file)
        # rotate the water bodies raster image
        wb_arr = np.rot90(wb_arr, k=-1)
        wb_mask = wb_arr == 2

        # Harmonize nctime calendar and units
        # SMAP time variables are selected as reference
        dc_smap.read()
        ref_unit = dc_smap.nc.variables['tb_time_utc'].units
        ref_calendar = dc_smap.nc.variables['tb_time_utc'].calendar
        dc_smap.close_nc()

        if os.path.exists(dc_cygn.fpath):
            succed, message = _copy_cygnss(dc, dc_cygn, ref_unit, ref_calendar, wb_mask)
            if not succed:
                raise Exception(message)
        else:
            print(ftile + '  not found! Most likely the tile is outside the CYGNSS coverage!')
        succed, message = _copy_smap(dc, dc_smap, ref_unit, ref_calendar, wb_mask)
        if not succed:
            raise Exception(message)

        #todo sort time series
        dc.open_nc()
        # append equi7grid lat/lon
        e7tile = Equi7Tile(ftile)
        e7grid = Equi7Grid(e7tile.res)
        size = dc.nc.dimensions['x'].size
        y_idx_arr = np.tile(np.array(range(size)), (size, 1))
        x_idx_arr = np.tile(np.array(range(size)), (size, 1)).T
        x_arr = x_idx_arr*e7tile.res + e7tile.llx + e7tile.res/2.0
        y_arr = y_idx_arr*e7tile.res + e7tile.lly + e7tile.res/2.0
        dc.nc['e7_lon'][:, :], dc.nc['e7_lat'][:, :] = e7grid.equi7xy2lonlat(ftile[0:2], x_arr, y_arr)
        dc.close_nc()
    
        # compress l2 netcdf file and move to destination directory
        os.makedirs(os.path.dirname(dc_dst.fpath), exist_ok=True)
        compress_netcdf(dc.fpath, dc_dst.fpath)
        logging.info(ftile + "  processed successfully! ..")

    except Exception as e:
        print(e)
        logging.error(ftile + "  processing failed! .." + "\n" + str(e))


def comb_ssm_l2u1_wrapper(dir_work, dir_dpool, dir_out=None, ftile_list=None, out_grid_res=None,
                          overwrite=False, mp_num=1):
    log_start_time = datetime.now()
    print(log_start_time, " COMB-SSM L2U1 data production started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_COMB-SSM_L2U1_data_production_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    if out_grid_res is None:
        out_grid_res = 6000
    elif int(out_grid_res) not in [3000, 6000]:
        raise ValueError("Given output grid resolution is not supported!")

    if ftile_list is None:
        ftiles = get_ftile_names(dir_dpool, grid_res=out_grid_res, land=True, eq=False)
    else:
        ftiles = ftile_list

    dir_cygnss_l2 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2")
    dir_smap_l3 = os.path.join(dir_dpool, "internal", "datacube", "smap_spl3smp_e", "dataset", "L3")
    dir_wb = os.path.join(dir_dpool, "internal", "datacube", "wb_esa_cci", "dataset")
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U1")

    if mp_num == 1:
        for ftile in ftiles:
            _gen_comb_ssm_l2u1(ftile, dir_work, dir_cygnss_l2, dir_smap_l3, dir_wb, dir_out, overwrite=overwrite)
    else:
        prod_ftile = partial(_gen_comb_ssm_l2u1, dir_work=dir_work, dir_cygnss_l2=dir_cygnss_l2,
                             dir_smap_l3=dir_smap_l3, dir_wb=dir_wb, dir_out=dir_out, overwrite=overwrite)
        p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)

    logging.info("============================================")
    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    logging.shutdown()
    print(datetime.now(), "COMB-SSM L2U1 data production is finished!")


@click.command()
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--out_grid_res', default=6000, type=int,
              help='if set different than grid_res, it will be used for output grid resolution')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(dir_work, dir_dpool, dir_out, out_grid_res, overwrite, mp_num):
    """
    This program generates COMB-SSM L2U1 data
    """
    ftile_list = None
    comb_ssm_l2u1_wrapper(dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list, out_grid_res=out_grid_res,
                          overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()
