import os
import pickle
import json
import glob
import traceback
import logging
import numpy as np
import click
import multiprocessing as mp
from functools import partial
from netCDF4 import Dataset
from datetime import datetime
from pygnssr.common.utils.Equi7Grid import Equi7Grid
from pygnssr.common.time.date_to_doys import date_to_ymd_str

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def spire_gnssr_latlon2e7_idx(file_in, dir_out, grid_res =3000, overwrite=False):
    """
    This program retrieves the Specular Point (SP) locations from Spire-GNSS-R netCDF file, searches for
    corresponding equi7grid point and stores the indices as Python object

    :param file_in: Spire-GNSS-R Level-1 netCDF data file
    :param grid_res: grid spacing (resolution) of target Equi7Grid
    :param dir_out: parent output directory
    :param overwrite: The outputfile, if exists, will be overwritten by setting to True
    :return: The Equi7 full-tile name, x and y indices of the corresponding point as python object file
    """

    try:
        date = os.path.basename(os.path.dirname(file_in))
        file_out = os.path.join(dir_out, date, os.path.basename(file_in) + '__e7indices'+'_'+str(grid_res)+ 'm.json')
        print(os.path.basename(file_in) + " data processing ....")
        if os.path.exists(file_out) and not overwrite:
            print("The file exists! " + str(datetime.now()) + "    " + os.path.basename(file_out))
        else:
            # read SP lon/lat arrays from cygnss netCDF file
            nc_in = Dataset(file_in, 'r')
            # get the point locations
            sp_lon = (nc_in.variables['sp_lon'][:].data[:]).astype('float64')
            # convert longitude in [0, 360] format to [-180°,180°]
            sp_lon = ((sp_lon - 180.0) % 360.0) - 180.0
            sp_lat = (nc_in.variables['sp_lat'][:].data[:]).astype('float64')
            nc_in.close()
            # Initialize equi7grid grid spacing
            grid = Equi7Grid(grid_res)
            # get the corresponding equi7 point locations
            ftile, _, _, ix, iy = grid.lonlat2equi7xy_idx(sp_lon, sp_lat)
            dic2dump = {'ftile': ftile.tolist(), 'ix': ix.tolist(), 'iy': iy.tolist()}
            # create sub-directory if not exists:
            os.makedirs(os.path.dirname(file_out), exist_ok=True)
            with open(file_out, "w") as f:
                json.dump(dic2dump, f)
            # print(os.path.basename(file_in)[0:21]+" data processing finished!")
            return None

    except Exception as e:
        message = traceback.format_exc()
        print(message)
        # remove output file if exists
        if os.path.exists(file_out):
            os.remove(file_out)
        print("Indexing failed! " + str(datetime.now()) + "    " + os.path.basename(file_out))
        return os.path.basename(file_in)


def _wrapper(dates_str_list, dir_in, dir_out, grid_res=3000, mp_num=1, overwrite=False):
    """
    This is a wrapper code for reading spire_gnssr files in given directory, searching for the nearest equi7grid points
    corresponding spire_gnssr observations and storing the indices as python object files in the given output directory

    :param dates_str_list: list of dates in following string format 'YYYY-MM-DD'
    :param dir_in: Input directory of spire_gnssr netCDF data files
    :param dir_out: Output directory to store indices as python object files
    :param mp_num: number of simultaneous processing instances (multiprocessing)
    :param overwrite: optional keyword. The output file will be overwritten if set as True
    """

    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_out,  datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    all_files_in =[]
    for date in dates_str_list:
        files_in = np.array(glob.glob(os.path.join(dir_in, '*', date, '*.nc')))
        if len(files_in) == 0:
            continue
        files_out = np.array([os.path.join(dir_out, date, os.path.basename(x) +
                                           '__e7indices'+'_'+str(grid_res)+'m.json') for x in files_in])

        filter_idx = []
        if not overwrite:
            for i, f in enumerate(files_out):
                if not os.path.exists(f):
                    filter_idx.append(i)
            if len(filter_idx) == 0:
                print(date+" data files were already processed (exist in output directory)!")
                continue
            files_in = files_in[filter_idx]
            all_files_in.extend(files_in)

    # create a pool of simultaneous processes
    partial_func = partial(spire_gnssr_latlon2e7_idx, dir_out=dir_out, grid_res=grid_res)
    not_proc_files = mp.Pool(processes=mp_num).map(partial_func, all_files_in)

    for f in not_proc_files:
        if f is not None:
            logging.error(f)

@click.command()
@click.option('--date', default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), type=str,
              help='Date in following format: "%Y-%m-%d %H:%M:%S"  Default is the current date.')
@click.option('--days', default=14, type=int,
              help='Number of days before the given date to be used as time filter for searching '
                   'and downloading the data files. Default is 14 days')
@click.option('-src', default=r"/home/ubuntu/datapool/internal/", type=str,
              help='Parent input directory. Default is the external datapool in "gnssr S3 bucket"')
@click.option('-dst', default=r"/home/ubuntu/datapool/internal/", type=str,
              help='Parent output directory. Default is the internal datapool in "gnssr S3 bucket"')
@click.option('-res', default=3000, type=int,
              help='Spatial resolution of target Equi7Grid in meters')
@click.option('-mp_num', default=16, type=int,
              help='Number of multiple processing')
def main(date, days, src, dst, res, mp_num):
    """
    This program finds the nearest Equi7Grid to CYGNSS observations and stores the results as indices in json file format.

    """
    print(datetime.now(), " Data indexing started from python code ...")
    dir_in = os.path.join(src, "spire_gnssr", "prod-0.3.7", "l1")
    dir_out = os.path.join(dst, "datacube", "spire_gnssr", "prod-0.3.7", "spire_gnssr_e7_indices")
    os.makedirs(dir_out, exist_ok=True)
    dates_str_list = date_to_ymd_str(date=datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), num_days=days)
    _wrapper(dates_str_list, dir_in, dir_out, grid_res=res, mp_num=mp_num)
    print(datetime.now(), "Data indexing is finished!")

if __name__ == "__main__":
 main()






