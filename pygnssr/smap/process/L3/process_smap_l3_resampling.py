import os
import shutil
import warnings
import click
import json
import pickle
import glob
import numpy as np
from datetime import datetime, date, timedelta
import logging
from netCDF4 import Dataset, num2date, date2num
from pygnssr.smap.SmapDataCube import SmapDataCube, get_l3_vars_template
import traceback
import multiprocessing as mp
from copy import deepcopy as cpy
import subprocess
import time
from pygnssr.common.utils.Equi7Grid import Equi7Grid, get_ftile_names
from functools import partial
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from pygnssr.smap.smap_utils import search_smap_h5_files, read_smap_subset
from matplotlib import pyplot as plt

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def resample_to_e7grid(ftile, files, dir_ease_e7_lut, dir_work, dir_out, update=False, mp_num=16, overwrite=False):


    """
    This program reads smap data-subset that overlaps with the given equi7 tile using the LUT between
    ease grid and e7grid

    :param ftile: Full tile name
    :param files: smap hdf5 full files path
    :param dir_ease_e7_lut: list of full index-files path
    :param dir_work: Working directory to store log files and intermediate data files
    :param dir_out: Output directory to store results
    :param update: If set True, then existing datacube will be updated
    :param mp_num: Number of files to read in parallel
    :param overwrite: If True, the dataset will be overwritten if exists in destiantion directory
    """
    try:
        # just to get the final output data path (no netcdf file is created or updated or read)
        dc_dst = SmapDataCube(ftile, 'L3', dir_out)
        if os.path.exists(dc_dst.fpath) and not (overwrite or update):
            raise ValueError("Output file exists! Set overwrite or update keyword as True..." + dc_dst.fpath)
        # make copy of h5 data files list
        files_2proc = cpy(files)

        if update and not os.path.exists(dc_dst.fpath):
            warnings.warn('No such file is available in output directory for update! ' + dc_dst.fpath)
            warnings.warn('A new file is created! ' + dc_dst.fpath)
            update = False

        # make an instance of datacube class
        dc = SmapDataCube(ftile, 'L3', dir_work)
        if update:
            try:
                # read the available datacube (netcdf file) to copy the variables and attributes
                dc_dst.read()
                # Avoid processing files that already exist in datacube, filter files
                proc_files = dc_dst.nc.variables['processed_files'][:]
                dc_dst.close_nc()
                if len(proc_files) != 0:
                    for f in files:
                        if os.path.basename(f) in proc_files:
                            files_2proc.remove(f)
                if len(files_2proc) == 0:
                    return True, 1
                else:
                    # copy dcube to working directory
                    # logging.info("Making a copy of dcube in working directory  " + ftile)
                    p = subprocess.Popen(['nccopy', dc_dst.fpath, dc.fpath])
                    p.communicate()
                    dc.open_nc()
                    dc_dst.read()
                    # update global attributes
                    dc.nc.history = getattr(dc.nc, 'history') # + '\n' + getattr(dc_dst.nc, 'history')
                    dc.close_nc()
                    dc_dst.close_nc()

            except OSError as e:
                print("Update error!..." + dc.fpath + "\n" + str(e))
                # logging.error("Update error:  " + dc.fpath)
                return None, 2
        else:
            dc = SmapDataCube(ftile, 'L3', dir_work, flag='w')
            # close netcdf file in working directory before reading input file using multiprocessing
            dc.close_nc()

        # logging.info("Data loading (reading) is started for tile: " + ftile)
        # logging.info("Number of input data files to be processed: " + str(len(files_2proc)))
        smap_vars = _read_smap_data(files_2proc, dir_ease_e7_lut=dir_ease_e7_lut, ftile=ftile, mp_num=mp_num)
        if smap_vars is None:
            logging.error("Files reading unsuccessful for tile:  " + ftile)
            os.remove(dc.fpath)
            return False, 2
        else:
            # logging.info("Data reading is finished! No. of successful reads: " + str(len(smap_vars['processed_files'])))
            print("Data reading is finished! No. of successful reads: " + str(len(smap_vars['processed_files'])))

        """
        # check the results
        var_name = 'bulk_density'
        for i in range(smap_vars[var_name].shape[0]):
            file_png = os.path.join('/home/ubuntu/_working_dir', str(i)+'.png')
            plt.imshow(smap_vars[var_name][i, :, :])
            plt.savefig(file_png)
            plt.close()
        """

        # open the netcdf file in working directory after the reading procedure
        dc.open_nc()

        # logging.info("Start transferring data array to netCDF ...")
        # get sizes of 'sample' and 'list' dimensions
        sample_size = dc.nc.dimensions['sample'].size
        list_size = dc.nc.dimensions['list'].size
        # ingest smap variables into the datacube
        # Since smap data are in raster format, opposite to GNSS-R data that are point measurements,
        # layers are added consecutively without any compression.

        for var_name in smap_vars.keys():
            if var_name == 'processed_files':
                dc.nc.variables[var_name][list_size:] = smap_vars[var_name][:]
            elif var_name == 'tb_time_utc':
                # convert datetime to netcdf4 date format (float number)
                t_obj = smap_vars['tb_time_utc']
                tt = np.ma.empty(t_obj.shape, dtype=np.float64, fill_value=-9999)
                tt.mask = t_obj.mask
                if len(t_obj[~t_obj.mask]) > 0:
                    tt[~t_obj.mask] = date2num(t_obj[~t_obj.mask],
                                               units=dc.nc.variables['tb_time_utc'].units,
                                               calendar=dc.nc.variables['tb_time_utc'].calendar)
                dc.nc.variables[var_name][sample_size:, :, :] = tt[:, :, :]
            else:
                dc.nc.variables[var_name][sample_size:, :, :] = smap_vars[var_name][:, :, :]

        # logging.info("Start writing data to disk...")
        dc.close_nc()
        # logging.info("Data writing is finished!")
        # logging.info("Start data compression ... ")

        os.makedirs(dir_out, exist_ok=True)
        # remove old data file if exists
        if os.path.exists(dc_dst.fpath):
            os.remove(dc_dst.fpath)

        smap_vars = 0
        tt = 0

        # compress the netcdf file and move to destination directory
        compress_netcdf(dc.fpath, dc_dst.fpath)

        # logging.info("Process is finished! ")
        return True, 3
    except OSError as e:
        return False, 4


def _read_smap_data(files, dir_ease_e7_lut=None, ftile=None, mp_num=1):
    col_lut, row_lut = _get_ease_e7_lut(dir_ease_e7_lut, ftile)
    try:
        #todo check if 'there is an overlapping with smap grid!')
        smap_vars = read_smap_subset(files, col_lut, row_lut, time_conversion=True, mp_num=mp_num)
        return smap_vars
    except Exception as e:
        message = traceback.format_exc()
        print(message)
        logging.error("Failed to process:  " + ftile)
        return None


def _get_ease_e7_lut(dir_ease_e7_lut, ftile):
    lut_file = os.path.join(dir_ease_e7_lut, ftile.split('_')[0], "ease_equi7_lut_"+ftile+".pkl")
    with open(lut_file, "rb") as f:
        col_lut, row_lut = pickle.load(f)
    return col_lut, row_lut


def _log_resample_to_e7grid(ftile, files, dir_ease_e7_lut, dir_work, dir_out, update=False, mp_num=16, overwrite=False):
    stime_ftile = datetime.now()
    try:
        dir_sub_out = os.path.join(dir_out, ftile.split('_')[0])
        succeed, pflag = resample_to_e7grid(ftile, files, dir_ease_e7_lut, dir_work, dir_sub_out,
                                            update=update, mp_num=mp_num, overwrite=overwrite)
        if succeed:
            logging.info("Sucessful data processing!" + ftile)
            if pflag == 1:
                logging.info("The given files have been already processed! Nothing to update for  " + ftile)
        else:
            logging.error("Tile processing error!..." + ftile)
    except OSError as e:
        logging.error("Tile processing error!..." + ftile + "\n" + str(e))
    logging.info("Total processing time for " + ftile+": "+str(datetime.now()-stime_ftile))
    logging.info("       ")


def smap_resampling_wrapper(dir_work, dir_dpool, dir_out=None, ftile_list=None, out_grid_res=None,
                            stime=None, etime=None, days=14, update=False, overwrite=False, mp_tiles=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, " SMAP Data resampling started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_SMAP_L3_dcube_creation_log_file.log")
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

    if ftile_list is None:
        ftiles = get_ftile_names(dir_dpool, grid_res=out_grid_res, land=True, eq=False)
    else:
        ftiles = ftile_list

    smap_dataset_name = "SPL3SMP_E"
    dir_smap = os.path.join(dir_dpool, "external", "smap", smap_dataset_name)
    dir_ease_e7_lut = os.path.join(dir_dpool, "internal", "misc", "ease_equi7_lut")
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "datacube", "smap_"+smap_dataset_name.lower(), "dataset", "L3")

    # search SMAP data files
    files = search_smap_h5_files(dir_smap, start_date=stime, end_date=etime)
    if len(files) == 0:
        print('No SMAP measurement found in the given time period!')
    else:
        if mp_tiles:
            prod_ftile = partial(_log_resample_to_e7grid, files=files, dir_ease_e7_lut=dir_ease_e7_lut,
                                 dir_work=dir_work, dir_out=dir_out, update=update, mp_num=1, overwrite=overwrite)
            p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)
        else:
            for ftile in ftiles:
                _log_resample_to_e7grid(ftile, files, dir_ease_e7_lut, dir_work, dir_out,
                                        update=update,  mp_num=mp_num, overwrite=overwrite)

    logging.info("============================================")
    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    logging.shutdown()
    print(datetime.now(), "SMAP data resampling is finished!")

@click.command()
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--out_grid_res', type=int,
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
@click.option('--update', is_flag=True,
              help='If set, the target datacube will be updated')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_tiles', is_flag=True,
              help='If set, the multi-processing will be applied over tiles rather than reading of input files')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(dir_work, dir_dpool, dir_out, out_grid_res, days, stime, etime, update, overwrite, mp_tiles, mp_num):
    """
    This program resamples(oversamples) SMAP 9km data to 3000m or 6000m Equi7grid
    """
    ftile_list = None
    smap_resampling_wrapper(dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list, out_grid_res=out_grid_res,
                            days=days, stime=stime, etime=etime, update=update, overwrite=overwrite,
                            mp_tiles=mp_tiles, mp_num=mp_num)


if __name__ == "__main__":
    main()