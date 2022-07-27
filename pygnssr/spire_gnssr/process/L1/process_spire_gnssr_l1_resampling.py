import os
import shutil
import warnings
import click
import json
import glob
import numpy as np
from datetime import datetime, date, timedelta
import logging
from netCDF4 import Dataset, num2date, date2num
from pygnssr.spire_gnssr.Spire_gnssrDataCube import Spire_gnssrDataCube, get_l1_vars_template
import traceback
import multiprocessing as mp
from copy import deepcopy as cpy
import subprocess
import time
from pygnssr.common.utils.Equi7Grid import Equi7Grid
from functools import partial
from pygnssr.common.utils.netcdf_utils import compress_netcdf

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def resample_to_e7grid(ftile, files, dir_spire_gnssr, dir_work, dir_out, update=False, mp_num=1, overwrite=False):
    """

    :param ftile: Full tile name
    :param files: Index files to be processed
    :param dir_spire_gnssr: Full directory path to spire_gnssr L1 data
    :param dir_work: Working directory to store log files and intermediate data files
    :param dir_out: Output directory
    :param update: If True, then existing datacube will be updated
    :param mp_num: Number of files to read in parallel
    :param overwrite: If True, the dataset will be overwritten if exists in destiantion directory
    """

    try:
        # just to get the final output data path (no netcdf file is created or updated or read)
        dc_dst = Spire_gnssrDataCube(ftile, 'L1', dir_out)
        if os.path.exists(dc_dst.fpath) and not (overwrite or update):
            raise ValueError("Output file exists! Set overwrite or update keyword as True..." +
                             os.path.basename(dc_dst.fpath))
        # make copy of ll idx files
        files_2proc = cpy(files)

        if update and not os.path.exists(dc_dst.fpath):
            warnings.warn('No such file is available in output directory for update! ' + dc_dst.fpath)
            warnings.warn('A new file is created! ' + dc_dst.fpath)
            update = False

        dc = Spire_gnssrDataCube(ftile, 'L1', dir_work)

        if update:
            try:
                # read the available datacube(netcdf file) to copy the variables and attributes
                dc_dst.read()
                # Avoid processing files that already exist in datacube, filter h5_files
                proc_files = dc_dst.nc.variables['processed_files'][:]
                dc_dst.close_nc()
                if len(proc_files) != 0:
                    for f in files:
                        if os.path.basename(f).split('__')[0] in proc_files:
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
                    dc.nc.history = 'Updated on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S') \
                                    +'\n' + getattr(dc_dst.nc, 'history')
                    # get the masked array of current netCDF file (need to find the last unmasked position
                    mask_arr = dc_dst.nc.variables['sp_lon'][:, :, :].mask
                    dc.close_nc()
                    dc_dst.close_nc()

            except Exception as e:
                print("Update error!..." + dc.fpath + "\n" + str(e))
                logging.error("Update error:  " + dc.fpath)
                return None, 2
        else:
            # TODO: select the sample file differently
            spire_gnssr_l1_sample_file = os.path.join(dir_spire_gnssr, "FM109", "2020-11-17",
                                                      "Spire_GNSSR_L1a_FM109_C0_s2020-11-17T00-00-00Z_e2020-11-17T01-00-00Z_v0.3.7.nc")

            # make an instance of datacube class
            dc = Spire_gnssrDataCube(ftile, 'L1', dir_work, flag='w', l1_sample_file=spire_gnssr_l1_sample_file)
            dc.close_nc()

        # get nctime units and calendar
        dc.read()
        units = dc.nc.variables['sample_time'].units
        calendar = dc.nc.variables['sample_time'].calendar
        dc.close_nc()


        # logging.info("Data loading (reading) is started for tile: " + ftile)
        # logging.info("Number of input data files to be processed: " + str(len(files_2proc)))
        if mp_num > 1:
            partial_func = partial(_read_spire_gnssr_subset, dir_spire_gnssr=dir_spire_gnssr, ftile=ftile, units=units, calendar=calendar)
            results = mp.Pool(processes=mp_num).map(partial_func, files_2proc)
        else:
            results = []
            for f in files_2proc:
                results.append(_read_spire_gnssr_subset(f, dir_spire_gnssr=dir_spire_gnssr, ftile=ftile, units=units, calendar=calendar))


        # logging.info("Data reading is finished! No. of successful reads:  " + str(len(np.where([r is not None for r in results])[0])))
        # logging.info("Start concatenating information read from the files ...")
        # initialize v1, and v2 arrays
        v1_arr, v3_arr= get_l1_vars_template()
        # concatenate indices from different files for given ftile
        init = True
        successful_read = False
        flist = []
        for k, file_idx in enumerate(files_2proc):
            fname = os.path.basename(file_idx).split("__")[0]
            # remove None results from the list
            if results[k] is None:
                continue
            flist.append(fname)
            if results[k] is False:
                continue
            successful_read = True
            if init:
                tx = cpy(results[k][0])
                ty = cpy(results[k][1])
                v1_arr = cpy(results[k][2])
                #v2_arr = cpy(results[k][3])
                init = False
            else:
                tx = np.concatenate((tx, results[k][0]))
                ty = np.concatenate((ty, results[k][1]))
                for var_name in v1_arr.keys():
                    v1_arr[var_name] = np.concatenate((v1_arr[var_name], results[k][2][var_name]))
                """
                for var_name in v2_arr.keys():
                    v2_arr[var_name] = np.concatenate((v2_arr[var_name], results[k][3][var_name]))
                """
            # free memory
            results[k] = None
        v3_arr['processed_files'] = np.array(flist)

        if successful_read:
            # logging.info("Sort time series at each pixel")
            xx = []
            yy = []
            idx = []
            ind_start = []
            ind_end = []
            dc.read()
            units = dc.nc.variables['sample_time'].units
            calendar = dc.nc.variables['sample_time'].calendar
            dc.close_nc()
            for x, y in set(zip(tx, ty)):
                xx.append(x)
                yy.append(y)
                t_arr = np.where((tx == x) & (ty == y))[0]
                # sort indices according to the acquisition time
                nc_time = v1_arr['sample_time'][t_arr]
                tt = [num2date(t, units, calendar=calendar) for t in nc_time]
                idx.extend([t_arr[np.argsort(tt)]])
                # initiate data collection with sample index equal to 0
                next_sample_ind = 0
                if update:
                    unmask_index = np.where(~mask_arr[:, x, y])[0]
                    # check if there is any masked position
                    if len(unmask_index) > 0:
                        next_sample_ind= unmask_index[-1] + 1

                ind_start.append(next_sample_ind)
                ind_end.append(next_sample_ind + len(t_arr))

            # logging.info("Start caching data in memory ...")
            sample_size = max(ind_end)
            if update:
                dc_dst.read()
                # get current sample size
                curr_sample_size = dc_dst.nc.dimensions['sample'].size
                # in case of update, curr_sample_size (already available samples) could be larger than input sample size
                # This is because the calculated sample size is done only for a subset of x, y
                cache_arr_1 = dc.get_l1_cache_vars(max(sample_size, curr_sample_size))
                for vname in dc.nc.variables.keys():
                    #TODO: NOTE that the updated datacube will not inlcude those field from
                    # destination dcube (the one that is being updated) that are not included
                    # in working datacube (they will simply be LOST after update!)

                    # check and copy the data field from the old datacube array if available
                    if vname in dc_dst.nc.variables.keys() and vname != 'processed_files':
                        cache_arr_1[vname][0:curr_sample_size, :, :] = dc_dst.nc[vname][:, :, :]

                # close destination datacube
                dc_dst.nc.close()
            else:
                #delay_size = 16 # v2_arr['power_reflect'].shape[1]
                #doppler_size = 23 # v2_arr['power_reflect'].shape[2]
                cache_arr_1= dc.get_l1_cache_vars(sample_size) #, delay_size, doppler_size)

            def _ts_cache_v1(px, py, pidx, pnum_start, pnum_end, vname, v1):
                cache_arr_1[vname][pnum_start:pnum_end, px, py] = v1[vname][pidx]
            """
            def _ts_cache_v2(px, py, pidx, pnum_start, pnum_end, vname, v2):
                cache_arr_2[vname][pnum_start:pnum_end, px, py, :, :] = v2[vname][pidx, :, :]
            """
            vfunc_v1 = np.vectorize(_ts_cache_v1, excluded=['vname', 'v1'])
            #vfunc_v2 = np.vectorize(_ts_cache_v2, excluded=['vname', 'v2'])

            # logging.info("Start transferring cached array to netCDF ...")
            # open the netcdf file in working directory to adapt
            dc.open_nc()

            for var_name in v1_arr.keys():
                vfunc_v1(px=xx, py=yy, pidx=idx, pnum_start=ind_start, pnum_end=ind_end, vname=var_name, v1=v1_arr)
                dc.nc.variables[var_name][:, :, :] = cache_arr_1[var_name][:, :, :]
            """
            for var_name in v2_arr.keys():
                vfunc_v2(px=xx, py=yy, pidx=idx, pnum_start=ind_start, pnum_end=ind_end, vname=var_name, v2=v2_arr)
                dc.nc.variables[var_name][:, :, :, :, :] = cache_arr_2[var_name][:, :, :, :, :]
            """
            dc.close_nc()
            # logging.info("Data writing is finished!")
            # logging.info("Start data compression ... ")
        else:
            # logging.info("No overlapping observation was found for this tile " + ftile)
            print("No overlapping observation was found for this tile " + ftile)

        if len(v3_arr['processed_files']) > 0:
            # logging.info("The list of processed files is being updated: " + ftile)
            # append processed file names
            dc.open_nc()
            list_size = dc.nc.dimensions['list'].size
            dc.nc.variables['processed_files'][list_size:] = v3_arr['processed_files'][:]
            # logging.info("Start writing data to disk...")
            dc.close_nc()
            os.makedirs(dir_out, exist_ok=True)
            # remove old data file if exists
            if os.path.exists(dc_dst.fpath):
                os.remove(dc_dst.fpath)
            # compress the netcdf file and move to destination directory
            compress_netcdf(dc.fpath, dc_dst.fpath)
            # logging.info("Process is finished! ")
        else:
            logging.info("The destination file is closed without any modification" + ftile)
            # remove DataCube file that was initiated in working directory
            os.remove(dc.fpath)

        return True, 3
    except Exception as e:
        logging.info(e)
        return False, 4


def _read_spire_gnssr_subset(file_idx, dir_spire_gnssr=None, ftile=None, units=None, calendar=None):
    try:
        if units is None or calendar is None:
            raise ValueError('units and calendar units need to be provided!')

        print("reading of " + os.path.basename(file_idx), datetime.now())
        sample_idx, ix, iy = _get_indices(file_idx, ftile)
        if len(np.where(sample_idx == True)[0]) < 1:
            print('No overlapping measurement!')
            return False
        v1, _ = get_l1_vars_template()
        # find corresponding spire_gnssr data file
        fname = os.path.basename(file_idx).split("__")[0]
        sat_name = fname.split("_")[3]
        obs_date = fname[26:36]
        nc_file_in = os.path.join(dir_spire_gnssr, sat_name, obs_date, fname)
        # open individual input spire_gnssr file
        nc_in = Dataset(nc_file_in, 'r')
        # read and concatenate variables type-1 from input files
        for var_name in v1.keys():
            if var_name == 'sample_time':
                # recalculate the date number as the date units could be different in each input data file
                date_nc_in = num2date(nc_in.variables[var_name][sample_idx].data,
                                      nc_in.variables[var_name].units, calendar=nc_in.variables[var_name].calendar)
                v1[var_name] = date2num(date_nc_in, units=units, calendar=calendar)

            else:
                v1[var_name] = nc_in.variables[var_name][sample_idx].data
        """
        # read and concatenate variables type-2 from input files
        for var_name in v2.keys():
            v2[var_name] = nc_in.variables[var_name][sample_idx, :, :].data
        """
        nc_in.close()
        return ix, iy, v1 #, v2
    except Exception as e:
        message = traceback.format_exc()
        print(message)
        logging.error("Failed to process:  " + file_idx)
        return None


def _get_indices(file_idx, ftile):
    with open(file_idx, "r") as f:
        # loading look-up-table between SMAP EASE grid and Equi7 grid
        dic = json.load(f)
    ft_arr = np.array(dic['ftile'])
    ix_arr = np.array(dic['ix'])
    iy_arr = np.array(dic['iy'])
    # todo: add an option to include overlapping tiles
    #  (by default the measurements are indexed only with one of the overlapping sgrids. )
    # indices matching the e7grid tile name
    sample_idx = ft_arr == ftile
    # sub set array of location indices
    ix = ix_arr[sample_idx]
    iy = iy_arr[sample_idx]
    return sample_idx, ix, iy


def _log_resample_to_e7grid(ftile, files, dir_spire_gnssr, dir_work, dir_out, update=False, mp_num=16, overwrite=False):
    stime_ftile = datetime.now()
    try:
        # define sub-directory using Equi7 sub-grid name
        dir_sub_out = os.path.join(dir_out, ftile.split('_')[0])
        succeed, pflag = resample_to_e7grid(ftile, files, dir_spire_gnssr, dir_work, dir_sub_out,
                                            update=update, mp_num=mp_num, overwrite=overwrite)
        if succeed:
            logging.info("Sucessful data processing!" + ftile)
            if pflag == 1:
                logging.info("The given files have been already processed! Nothing to update for  " + ftile)
        else:
            logging.error("Tile processing error!..." + ftile)
    except Exception as e:
        logging.error("Tile processing error!..." + ftile + "\n" + str(e))
    logging.info("Total processing time for " + ftile+": "+str(datetime.now()-stime_ftile))
    logging.info("       ")


def spire_gnssr_resampling_wrapper(grid_res, dir_work, dir_dpool, dir_out=None, ftile_list=None, stime=None, etime=None,
                                   days=14, update=False, overwrite=False, mp_tiles=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, " Data resampling started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_spire_gnssr_L1_dcube_creation_log_file.log")
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

    if ftile_list is None:
        ftiles = get_ftile_names(dir_dpool, grid_res=grid_res, land=True, eq=True)
    else:
        ftiles = ftile_list

    dir_spire_gnssr = os.path.join(dir_dpool, "internal", "spire_gnssr", "prod-0.3.7", "l1")
    dir_e7grid_idx = os.path.join(dir_dpool, "internal", "datacube", "spire_gnssr", "prod-0.3.7", "spire_gnssr_e7_indices")
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "datacube", "spire_gnssr", "prod-0.3.7", "dataset", "L1")
    all_files = glob.glob(os.path.join(dir_e7grid_idx, '*/Spire_GNSSR*.json'))
    dates = np.array([datetime.strptime(os.path.basename(x)[26:36], "%Y-%m-%d") for x in all_files])

    # filter dates
    if (stime is not None) and (etime is not None):
        ind = np.where((dates >= stime) & (dates <= etime))
        files = list(np.array(all_files)[ind])
    if len(files) == 0:
        print('No CYGNSS measurement found in the given time period!')
    else:
        if mp_tiles:
            prod_ftile = partial(_log_resample_to_e7grid, files=files, dir_spire_gnssr=dir_spire_gnssr, dir_work=dir_work,
                                 dir_out=dir_out, update=update, mp_num=1, overwrite=overwrite)
            p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)
        else:
            for ftile in ftiles:
                _log_resample_to_e7grid(ftile, files, dir_spire_gnssr, dir_work, dir_out,
                                        update=update, mp_num=mp_num, overwrite=overwrite)

    logging.info("============================================")
    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    print(datetime.now(), "spire_gnssr data resampling is finished!")


@click.command()
@click.argument('grid_res', default='3000', type=str)
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
@click.option('--update', is_flag=True,
              help='If set, the target datacube will be updated')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_tiles', is_flag=True,
              help='If set, the multi-processing will be applied over tiles rather than reading of input files')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dir_work, dir_dpool, dir_out, days, stime, etime, update, overwrite, mp_tiles, mp_num):
    """
    This program resamples spire_gnssr level-1 data to Equi7Grid and creates/updates spire_gnssr Data Cube

    :param grid_res: grid spacing, default is 3000 meter
    """
    ftile_list = None
    days = 300
    mp_tiles = True
    spire_gnssr_resampling_wrapper(grid_res, dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list, stime=stime,
                                   etime=etime, days=days, update=update, overwrite=overwrite, mp_tiles=mp_tiles, mp_num=mp_num)


if __name__ == "__main__":
    main()
