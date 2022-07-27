import os
import click
import json
import glob
import logging
import warnings
import traceback
import numpy as np
from datetime import datetime, date, timedelta
from netCDF4 import Dataset, num2date, date2num
from copy import deepcopy as cpy
import subprocess
import multiprocessing as mp
from functools import partial
from pygnssr.common.utils.Equi7Grid import Equi7Grid
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from pygnssr.sim_gnssr.SimDataCube import SimDataCube, get_l1_vars_template

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def resample_to_e7grid(ftile, files, dir_sim_gnssr, dir_work, dir_out, update=False, mp_num=1, overwrite=False):
    """
    This program reads spire_gnssr data-subset that overlaps with the given equi7 tile using the index arrays between
    spire_gnssr and e7grid

    :param files_idx: list of full paths to index-files
    :param dir_work: Working directory to store log files and intermediate data files
    :param dir_out: Output directory to store results
    :param update: If set True, then existing datacube will be updated
    :param num_process: Number of files to read in parallel

    :return:
    """

    try:
        # just to get the final output data path (no netcdf file is created or updated or read)
        dc_dst = SimDataCube(ftile, 'L1', dir_out)
        if os.path.exists(dc_dst.fpath) and not (overwrite or update):
            raise ValueError("Output file exists! Set overwrite or update keyword as True..." + dc_dst.fpath)
        # make copy of ll idx files
        files_2proc = cpy(files)

        if update and not os.path.exists(dc_dst.fpath):
            warnings.warn('No such file is available in output directory for update! ' + dc_dst.fpath)
            warnings.warn('A new file is created! ' + dc_dst.fpath)
            update = False

        dc = SimDataCube(ftile, 'L1', dir_work)
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
                    dc.nc.history = getattr(dc.nc, 'history')  # + '\n' + getattr(dc_dst.nc, 'history')
                    # get the masked array of current netCDF file (need to find the last unmasked position
                    mask_arr = np.ma.getmaskarray(dc_dst.nc.variables['sp_lon'])
                    dc.close_nc()
                    dc_dst.close_nc()

            except OSError as e:
                print("Update error!..." + dc.fpath + "\n" + str(e))
                # logging.error("Update error:  " + dc.fpath)
                return None, 2
        else:
            # make an instance of datacube class
            dc = SimDataCube(ftile, 'L1', dir_work, flag='w')
            dc.close_nc()

        # get nctime units and calendar
        dc.read()
        units = dc.nc.variables['sample_time'].units
        calendar = dc.nc.variables['sample_time'].calendar
        dc.close_nc()
        logging.info("Data loading (reading) is started for tile: " + ftile)
        logging.info("Number of input data files to be processed: " + str(len(files_2proc)))
        if mp_num > 1:
            partial_func = partial(_read_sim_gnssr_subset, dir_sim_gnssr=dir_sim_gnssr, ftile=ftile, units=units,
                                   calendar=calendar)
            results = mp.Pool(processes=mp_num).map(partial_func, files_2proc)
        else:
            results = []
            for f in files_2proc:
                results.append(
                    _read_sim_gnssr_subset(f, dir_sim_gnssr=dir_sim_gnssr, ftile=ftile, units=units, calendar=calendar))

        logging.info("Data reading is finished! No. of successful reads:  " + str(len(np.where([r is not None for r in results])[0])))
        logging.info("Start concatenating information read from the files ...")
        # initialize v1, v2, and v3 arrays
        v_arr = get_l1_vars_template()
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
                v_arr = cpy(results[k][2])
                init = False
            else:
                tx = np.concatenate((tx, results[k][0]))
                ty = np.concatenate((ty, results[k][1]))
                for var_name in v_arr.keys():
                    if var_name != 'processed_files':
                        v_arr[var_name] = np.concatenate((v_arr[var_name], results[k][2][var_name]))

            # free memory
            results[k] = None
        v_arr['processed_files'] = np.array(flist)

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
                nc_time = v_arr['sample_time'][t_arr]
                tt = [num2date(t, units, calendar=calendar) for t in nc_time]
                idx.extend([t_arr[np.argsort(tt)]])
                # initiate data collection with sample index equal to 0
                next_sample_ind = 0
                if update:
                    unmask_index = np.where(~mask_arr[:, x, y])[0]
                    # check if there is any masked position
                    if len(unmask_index) > 0:
                        next_sample_ind = unmask_index[-1] + 1

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
                cache_arr = dc.get_l1_cache_vars(max(sample_size, curr_sample_size))
                for vname in dc.nc.variables.keys():
                    # check and copy the data field from the old datacube array if available
                    if vname in dc_dst.nc.variables.keys() and vname != 'processed_files':
                        cache_arr[vname][0:curr_sample_size, :, :] = dc_dst.nc[vname][:, :, :]

                # close destination datacube
                dc_dst.nc.close()
            else:
                cache_arr = dc.get_l1_cache_vars(sample_size)

            def _ts_cache_v(px, py, pidx, pnum_start, pnum_end, vname, v):
                cache_arr[vname][pnum_start:pnum_end, px, py] = v[vname][pidx]
            vfunc_v = np.vectorize(_ts_cache_v, excluded=['vname', 'v'])

            logging.info("Start transferring cached array to netCDF ...")
            # open the netcdf file in working directory to adapt
            dc.open_nc()
            for var_name in v_arr.keys():
                if var_name not in ['processed_files', 'tx_system', 'sat_name']:
                    vfunc_v(px=xx, py=yy, pidx=idx, pnum_start=ind_start, pnum_end=ind_end, vname=var_name, v=v_arr)
                    dc.nc.variables[var_name][:, :, :] = cache_arr[var_name][:, :, :]
                if var_name in ['tx_system', 'sat_name']:
                    vfunc_v(px=xx, py=yy, pidx=idx, pnum_start=ind_start, pnum_end=ind_end, vname=var_name, v=v_arr)
                    dc.nc.variables[var_name][:, :, :] = cache_arr[var_name].filled("")[:, :, :]

            dc.close_nc()
            logging.info("Data writing is finished!")
            logging.info("Start data compression ... ")

        else:
            # logging.info("No overlapping observation was found for this tile " + ftile)
            print("No overlapping observation was found for this tile " + ftile)

        if len(v_arr['processed_files']) > 0:
            # logging.info("The list of processed files is being updated: " + ftile)
            # append processed file names
            dc.open_nc()
            list_size = dc.nc.dimensions['list'].size
            dc.nc.variables['processed_files'][list_size:] = v_arr['processed_files'][:]
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
    except OSError as e:
        return False, 4


def _read_sim_gnssr_subset(file_idx, dir_sim_gnssr=None, ftile=None, units=None, calendar=None):
    try:
        if units is None or calendar is None:
            raise ValueError('units and calendar units need to be provided!')

        print("reading of " + os.path.basename(file_idx), datetime.now())
        sample_idx, ix, iy = _get_indices(file_idx, ftile)
        val_num = len(np.where(sample_idx == True)[0])
        if val_num == 0:
            print('No overlapping measurement!')
            return False
        v = get_l1_vars_template()
        # find corresponding spire_gnssr data file
        fname = os.path.basename(file_idx).split("__")[0]
        a = fname.index('_') + 1
        b = fname.index('_time_')
        sat_name = fname[a:b]
        sim_file_in = os.path.join(dir_sim_gnssr, fname)
        # open individual input spire_gnssr file
        nc_in = Dataset(sim_file_in, 'r')
        # read and concatenate variables type-1 from input files
        for var_name in v.keys():
            if var_name == 'sample_time':
                # recalculate the date number as the date units could be different in each input data file
                date_nc_in = num2date(nc_in.variables[var_name][sample_idx].data,
                                      nc_in.variables[var_name].units, calendar=nc_in.variables[var_name].calendar)
                v[var_name] = date2num(date_nc_in, units=units, calendar=calendar)
            elif var_name == 'sat_name':
                v[var_name] = np.repeat(sat_name, val_num)
            elif var_name == 'tx_system':
                v[var_name] = np.ma.array(nc_in.variables[var_name])[sample_idx].data
            elif var_name == 'processed_files':
                continue
            else:
                v[var_name] = nc_in.variables[var_name][sample_idx].data
        nc_in.close()
        return ix, iy, v
    except Exception as e:
        message = traceback.format_exc()
        print(message)
        logging.error("Failed to process:  " + file_idx)
        return None


def _get_indices(file_idx, ftile):
    with open(file_idx, "r") as f:
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


def _log_resample_to_e7grid(ftile, files, dir_sim_gnssr, dir_work, dir_out, update=False, mp_num=16, overwrite=False):
    stime_ftile = datetime.now()
    try:
        # define sub-directory using Equi7 sub-grid name
        dir_sub_out = os.path.join(dir_out, ftile.split('_')[0])
        succeed, pflag = resample_to_e7grid(ftile, files, dir_sim_gnssr, dir_work, dir_sub_out,
                                            update=update, mp_num=mp_num, overwrite=overwrite)
        if succeed:
            logging.info("Sucessful data processing!" + ftile)
            if pflag == 1:
                logging.info("The given files have been already processed! Nothing to update for  " + ftile)
        else:
            logging.error("Tile processing error!..." + ftile)
    except OSError as e:
        logging.error("Tile processing error!..." + ftile + "\n" + str(e))
    logging.info("Total processing time for " + ftile + ": " + str(datetime.now() - stime_ftile))
    logging.info("       ")


def sim_resampling_wrapper(grid_res, dir_work, dir_dpool, dir_out=None, ftile_list=None,
                           stime=None, etime=None, days=14, update=False, overwrite=False, mp_tiles=False, mp_num=1, node=None):
    log_start_time = datetime.now()
    print(log_start_time, " Simulate Data resampling started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_SIM_dcube_creation_log_file.log")
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
        ftiles = get_ftile_names(dir_dpool, grid_res=grid_res, land=True, eq=False)
    else:
        ftiles = ftile_list

    if node is not None:
        a = (node-1) * 5
        b = min([a+5, len(ftiles)])
        ftiles = ftiles[a:b]

    if grid_res == 1000:
        in_data_name = "schedules_att90_netcdf_int_10_hz"
        out_data_name = "sim_gnssr_10hz"
    else:
        in_data_name = "schedules_att90_netcdf_int_2_hz"
        out_data_name = "sim_gnssr_2hz"

    dir_sim_gnssr = os.path.join(dir_dpool, "internal", "temp_working_dir", "2020-09-17_gnss-r_coverage_maps",
                                 "schedules_att90_netcdf_int", in_data_name)
    dir_e7grid_idx = os.path.join(dir_dpool, "internal", "datacube", out_data_name, "sim_gnssr_e7_indices",
                                  "G" + str(grid_res) + "M")
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "datacube", out_data_name, "dataset")
    files = glob.glob(os.path.join(dir_e7grid_idx, 'sched_*.json'))

    if len(files) == 0:
        print('No sim_gnssr measurement found!')
    else:
        if mp_tiles:
            prod_ftile = partial(_log_resample_to_e7grid, files=files, dir_sim_gnssr=dir_sim_gnssr, dir_work=dir_work,
                                 dir_out=dir_out, update=update, mp_num=1, overwrite=overwrite)
            p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)
        else:
            for ftile in ftiles:
                _log_resample_to_e7grid(ftile, files, dir_sim_gnssr, dir_work, dir_out,
                                        update=update, mp_num=mp_num, overwrite=overwrite)

    logging.info("============================================")
    logging.info("Total processing time " + str(datetime.now() - log_start_time))
    print(datetime.now(), "SIM_gnssr data resampling is finished!")


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
@click.option('--mp_num', default=1, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dir_work, dir_dpool, dir_out, days, stime, etime, update, overwrite, mp_tiles, mp_num):
    """
    This program resamples simulated GNSS-R data to Equi7Grid and creates/updates Data Cube

    :param grid_res: grid spacing, default is 3000 meter
    """
    ftile_list = None
    sim_resampling_wrapper(grid_res, dir_work, dir_dpool, dir_out=dir_out, ftile_list=None, stime=stime, etime=etime,
                           days=days, update=update, overwrite=overwrite, mp_tiles=mp_tiles, mp_num=mp_num, node=node)


if __name__ == "__main__":
    main()
