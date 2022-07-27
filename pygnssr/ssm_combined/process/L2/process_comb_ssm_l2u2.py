import os
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.common.utils.netcdf_utils import compress_netcdf
import json
from functools import partial
import multiprocessing as mp
from pygnssr.ssm_combined.CombSSMDataCube import CombSSMDataCube
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
import logging
from datetime import datetime, timedelta
import glob
import click

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _gen_comb_ssm_l2u2(ftile, dir_in, dir_out, dir_work=None, start_date=None, end_date=None, int_type=None,
                       date_int=None, bbox=None, overwrite=False):
    try:
        if dir_work is None:
            dir_work = dir_out
        if date_int is None:
            date_int = get_time_intervals(start_date, end_date, interval_type=int_type)

        e7tile = Equi7Tile(ftile)
        e7grid = Equi7Grid(e7tile.res)
        if bbox is not None:
            ftile_min, x_min, y_min, ix_min, iy_min = e7grid.lonlat2equi7xy_idx(bbox[0], bbox[1])
            ftile_max, x_max, y_max, ix_max, iy_max = e7grid.lonlat2equi7xy_idx(bbox[2], bbox[3])
            if ftile_min != ftile:
                if x_min < e7tile.llx:
                    ix_min = 0
                if y_min < e7tile.lly:
                    iy_min = 0
                if x_min > (e7tile.llx + e7tile._xspan) or y_min > (e7tile.lly + e7tile._yspan):
                    raise ValueError("Given boundary box is outside the tile")
            if ftile_max != ftile:
                if x_max > (e7tile.llx + e7tile._xspan):
                    ix_max = -1
                if y_max > (e7tile.lly + e7tile._yspan):
                    iy_max = -1
                if x_max < e7tile.llx  or y_max < e7tile.lly:
                    raise ValueError("Given boundary box is outside the tile")
            bx = [int(ix_min), int(iy_min), int(ix_max), int(iy_max)]
        else:
            bx = [0, 0, -1, -1]

        dc_src = CombSSMDataCube(ftile, "L2U1", os.path.join(dir_in, e7tile.sgrid), flag='r')
        smap_time = dc_src.nc['smap']['time_utc']
        smap_un = dc_src.nc['smap']['time_utc'].units
        smap_cal = dc_src.nc['smap']['time_utc'].calendar
        cygn_time = dc_src.nc['cygnss']['time_utc']
        cygn_un = dc_src.nc['cygnss']['time_utc'].units
        cygn_cal = dc_src.nc['cygnss']['time_utc'].calendar

        dir_sub_out = os.path.join(dir_out, e7tile.sgrid, e7tile.tilename)
        os.makedirs(dir_sub_out, exist_ok=True)
        for st, et in zip(date_int[0], date_int[1]):
            file_dst = os.path.join(dir_sub_out, "COMB-SSM_L2U2_" + st.strftime("%Y%m%dT%H%M%S") + "_"
                                    + (et-timedelta(microseconds=1)).strftime("%Y%m%dT%H%M%S")
                                    + "_" + ftile + ".nc")
            if os.path.exists(file_dst) and not overwrite:
                print("File exists in destination directory! Set overwrite keyword!", file_dst)
                continue

            # get valid indices within the given time period
            val_smap = np.ma.masked_greater_equal(np.ma.masked_less
                                                  (smap_time, date2num(st, units=smap_un, calendar=smap_cal)),
                                                  date2num(et, units=smap_un, calendar=smap_cal))
            val_cygn = np.ma.masked_greater_equal(np.ma.masked_less
                                                  (cygn_time, date2num(st, units=cygn_un, calendar=cygn_cal)),
                                                  date2num(et, units=cygn_un, calendar=cygn_cal))

            if val_smap.count() == 0 and val_cygn.count() == 0:
                continue
            dc = CombSSMDataCube(ftile, "L2U2", dir_work, flag='w')
            dc.nc.variables['e7_lon'][:, :] = dc_src.nc.variables['e7_lon'][:, :]
            dc.nc.variables['e7_lat'][:, :] = dc_src.nc.variables['e7_lat'][:, :]

            if val_smap.count() != 0:
                # mask variables
                smap_t = np.ma.masked_where(np.ma.getmask(val_smap), dc_src.nc['smap']['time_utc'])
                smap_sm = np.ma.masked_where(np.ma.getmask(val_smap), dc_src.nc['smap']['sm'])
                smap_qflag = np.ma.masked_where(np.ma.getmask(val_smap), dc_src.nc['smap']['retrieval_qual_flag'])
                smap_sflag = np.ma.masked_where(np.ma.getmask(val_smap), dc_src.nc['smap']['surface_flag'])
                # write valid slices in netcdf file
                t_dst = 0
                for t_src in range(smap_t.shape[0]):
                    if smap_sm[t_src, bx[0]:bx[2], bx[1]:bx[3]].all() is not np.ma.masked:
                        dc.nc['smap']['time_utc'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = smap_t[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['smap']['sm'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = smap_sm[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['smap']['retrieval_qual_flag'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = smap_qflag[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['smap']['surface_flag'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = smap_sflag[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        t_dst = t_dst + 1
            
            if val_cygn.count() != 0:
                # mask variables
                cygn_t = np.ma.masked_where(np.ma.getmask(val_cygn), dc_src.nc['cygnss']['time_utc'])
                cygn_rssm = np.ma.masked_where(np.ma.getmask(val_cygn), dc_src.nc['cygnss']['rssm'])
                cygn_cssm = np.ma.masked_where(np.ma.getmask(val_cygn), dc_src.nc['cygnss']['cssm'])
                cygn_sp_lon = np.ma.masked_where(np.ma.getmask(val_cygn), dc_src.nc['cygnss']['sp_lon'])
                cygn_sp_lat = np.ma.masked_where(np.ma.getmask(val_cygn), dc_src.nc['cygnss']['sp_lat'])
                # write valid slices in netcdf file
                t_dst = 0
                for t_src in range(cygn_t.shape[0]):
                    if cygn_rssm[t_src, bx[0]:bx[2], bx[1]:bx[3]].all() is not np.ma.masked:
                        dc.nc['cygnss']['time_utc'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = cygn_t[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['cygnss']['rssm'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = cygn_rssm[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['cygnss']['cssm'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = cygn_cssm[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['cygnss']['sp_lon'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = cygn_sp_lon[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        dc.nc['cygnss']['sp_lat'][t_dst, bx[0]:bx[2], bx[1]:bx[3]] = cygn_sp_lat[t_src, bx[0]:bx[2], bx[1]:bx[3]]
                        t_dst = t_dst + 1

            file_src = dc.nc.filepath()
            # close net
            dc.close_nc()
            # compress the netcdf file and move to destination directory
            compress_netcdf(file_src, file_dst)
            logging.info(ftile + "  processed successfully! ..")

    except Exception as e:
        print("Failed to process:  " + ftile, e)
        logging.error(ftile + "  processing failed! .." + "\n" + str(e))
        return None


def comb_ssm_l2u2_wrapper(grid_res, dir_work, dir_dpool, dir_out=None, ftile_list=None, stime=None, etime=None,
                          days=14, int_type='6h', overwrite=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, " COMB-SSM L2U2 data production started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_COMB-SSM_L2U2_data_production_log_file.log")
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

    date_int = get_time_intervals(stime, etime, interval_type=int_type)

    if ftile_list is None:
        ftiles = get_ftile_names(dir_dpool, grid_res=grid_res, land=True, eq=False)
    else:
        ftiles = ftile_list

    dir_comb_ssm_l2u1 = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U1")
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U2")

    if mp_num == 1:
        for ftile in ftiles:
            _gen_comb_ssm_l2u2(ftile, dir_comb_ssm_l2u1, dir_out,
                               dir_work=dir_work, date_int=date_int, overwrite=overwrite)
    else:
        prod_ftile = partial(_gen_comb_ssm_l2u2, dir_in=dir_comb_ssm_l2u1, dir_out=dir_out,
                             dir_work=dir_work, date_int=date_int, overwrite=overwrite)
        p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)

    logging.info("============================================")
    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    logging.shutdown()
    print(datetime.now(), "COMB-SSM L2U2 data production is finished!")


@click.command()
@click.argument('grid_res', default='6000', type=str)
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--days', default=14, type=int,
              help='Number of days in the past to be used as time filter for searching '
                   'and downloading the data files. Default is 14 days')
@click.option('--stime', default=None, type=str,
              help='Start date and time in following format: "%Y-%m-%d %H:%M:%S"  Default is None. '
                   'If provided, then the "days" argument will be overridden!')
@click.option('--etime', default=None, type=str,
              help='End date and time in following format: "%Y-%m-%d %H:%M:%S"  Default is None.'
                   'If not provided, then current date will be used as end time')
@click.option('--int_type', default='6h', type=str,
              help='Time interpolation type. Default is 6 hours interval')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dir_work, dir_dpool, dir_out, days, stime, etime, int_type, overwrite, mp_num):
    """
    This program generates COMB-SSM L2U1 data
    """
    ftile_list = None
    comb_ssm_l2u2_wrapper(grid_res, dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list,
                          stime=stime, etime=etime, days=days, int_type=int_type, overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()




