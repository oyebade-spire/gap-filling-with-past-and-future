import os
import click
import json
import glob
import logging
import warnings
import traceback
import numpy as np
from datetime import datetime, date, timedelta
from time import sleep
from netCDF4 import Dataset, num2date, date2num
from copy import deepcopy as cpy
import subprocess
import multiprocessing as mp
from functools import partial
from pygnssr.common.utils.Equi7Grid import Equi7Grid
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.cygnss.process.L1.process_cygnss_l1_resampling import cygnss_resampling_wrapper
from pygnssr.cygnss.process.L2.process_cygnss_l2_sm import cygnss_sm_wrapper
from pygnssr.smap.process.L3.process_smap_l3_resampling import smap_resampling_wrapper
from pygnssr.ssm_combined.process.L2.process_comb_ssm_l2u1 import comb_ssm_l2u1_wrapper
from pygnssr.ssm_combined.process.L2.process_comb_ssm_l2u2 import comb_ssm_l2u2_wrapper
from pygnssr.ssm_combined.process.L3.process_comb_ssm_mean_l3u1 import comb_ssm_l3u1_wrapper


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _exist_l3u1_files(start_time, end_time, dir_l3u1, grid_res):
    """
    :param start_time: starting the time period
    :param end_time: ending the time period
    :param dir_l3u1: directory of L3U1 files
    :param grid_res: grid spacing
    :return: start times of non-existing l3u1 files
    """
    date_int = get_time_intervals(start_time, end_time, interval_type='daily')
    # output file names
    start_times = []
    for st, et in zip(date_int[0], date_int[1]):
        name = st.strftime("%Y%m%d") + "_" + et.strftime("%Y%m%d") + "_COMB-SSM_L3U1_"+str(grid_res)+"M.tif"
        if not os.path.exists(os.path.join(dir_l3u1, name)):
            start_times.append(st)
    return start_times


def main():
    log_start_time = datetime.now()
    iteration_start = log_start_time
    print(log_start_time, " COMB-SSM NRT workflow started from python code ...")

    dir_work = r"/home/ubuntu/_working_dir"
    dir_dpool = r"/home/ubuntu/datapool"
    dir_l1 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L1")
    dir_e7grid_idx = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "cygnss_e7_indices")
    dir_l2 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2")
    dir_smap = os.path.join(dir_dpool, "internal", "datacube", "smap_spl3smp_e", "dataset", "L3")
    dir_comb_ssm_L2u1 = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U1")
    dir_comb_ssm_L2u2 = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U2")
    dir_comb_ssm_L3u1 = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L3U1")
    pygnssr_path = r"/home/ubuntu/swdvlp/python/pygnssr/pygnssr"
    l1_grid_res = 3000
    l2_grid_res = 6000
    days_num = 14  # number of days to consider for data processing
    latency_days = 7  # the files older than latency_days will be processed regardless of completeness

    while True:
        end_time = datetime.now()
        #end_time = datetime(2020, 3, 1)
        start_time = end_time - timedelta(days_num)
        # start_time = datetime(2020, 10, 19)

        # check output files (check if l3u1 files already exist)
        start_times = _exist_l3u1_files(start_time, end_time, dir_comb_ssm_L3u1, l2_grid_res)
        if len(start_times) != 0:
            # Check input files (check if cygnss index files already exist)
            stimes = []
            for st in start_times:
                files = glob.glob(os.path.join(dir_e7grid_idx, str(st.year),
                                               "cyg*.ddmi.s"+st.strftime("%Y%m%d")+"-*.json"))
                if len(files) == 8 or (datetime.now() - st).days > latency_days:
                    stimes.append(st)
                else:
                    print(st.strftime("%Y%m%d") + "  Not enough files available!  file_nums:  " + str(len(files)))
            if len(stimes) != 0:
                stime = min(stimes)
                a = max(stimes)
                etime = datetime(a.year, a.month, a.day)+timedelta(days=1)

                #  CYGNSS Resampling 3000M  ------------------------------------------------------------------------
                # ftile_list is the list of updated ftiles
                ftile_list = cygnss_resampling_wrapper(str(l1_grid_res), dir_work, dir_dpool, dir_out=dir_l1,
                                                             stime=stime.strftime("%Y-%m-%d %H:%M:%S"),
                                                             etime=etime.strftime("%Y-%m-%d %H:%M:%S"),
                                                             update=True, mp_tiles=True, mp_num=4)
                print(ftile_list)
                if ftile_list is not None:
                    #  CYGNSS SSM calculation 6000M ----------------------------------------------------------------
                    cygnss_sm_wrapper(str(l1_grid_res), dir_work, dir_dpool, dir_out=dir_l2, ftile_list=ftile_list,
                                      out_grid_res=str(l2_grid_res), overwrite=True, mp_num=8)

                    #  SMAP SSM resampling 6000M -------------------------------------------------------------------
                    smap_resampling_wrapper(dir_work, dir_dpool, dir_out=dir_smap, out_grid_res=str(l2_grid_res),
                                            stime=stime.strftime("%Y-%m-%d %H:%M:%S"),
                                            etime=etime.strftime("%Y-%m-%d %H:%M:%S"),
                                            update=True, mp_tiles=True, mp_num=8)

                    #  COMB-SSM L2U1  ----  merging CYGNSS and SMAP to one datacube --------------------------------
                    comb_ssm_l2u1_wrapper(dir_work, dir_dpool, dir_out=dir_comb_ssm_L2u1,
                                          out_grid_res=str(l2_grid_res),
                                          overwrite=True, mp_num=8)

                    #  COMB-SSM L2U2  ----  splitting L2U1 to 6 hourly product -------------------------------------
                    comb_ssm_l2u2_wrapper(str(l2_grid_res), dir_work, dir_dpool, dir_out=dir_comb_ssm_L2u2,
                                          stime=stime.strftime("%Y-%m-%d %H:%M:%S"),
                                          etime=etime.strftime("%Y-%m-%d %H:%M:%S"),
                                          int_type='6h', mp_num=6)

                    #  COMB-SSM L3U1  ----  making daily geotiff images from sMAP and CYGNSS SSM--------------------
                    comb_ssm_l3u1_wrapper(dir_work, dir_dpool, dir_out=dir_comb_ssm_L3u1,
                                          out_grid_res=str(l2_grid_res),
                                          stime=stime.strftime("%Y-%m-%d %H:%M:%S"),
                                          etime=etime.strftime("%Y-%m-%d %H:%M:%S"),
                                          int_type='daily', mp_num=7)

            else:
                print("No new dataset available!  Resampling is not performed!")

        # trigger the next iteration
        iteration_start = iteration_start.replace(hour=0, minute=0, second=0, microsecond=0)+timedelta(days=1)
        while datetime.now() < iteration_start:
            sleep(600)


if __name__ == "__main__":
    main()
