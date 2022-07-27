from netCDF4 import Dataset, num2date, date2num
import numpy as np
import os
import glob
import copy
from datetime import datetime, timedelta
from itertools import repeat
import pyproj
import subprocess
import multiprocessing as mp
from functools import partial
from collections import OrderedDict
from pygnssr.common.utils.netcdf_utils import compress_netcdf

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


def interpolate_obs(file_in, dir_out, dir_work, sampling_rate =1, overwrite=False):
    """
    This program retrieves the Specular Point (SP) locations from GNSS-R coverage file, searches for
    corresponding equi7grid point and stores the indices as Python object in given output directory

    :param file_in: GNSS-R json data file
    :param file_out: full path of output file
    :param sampling_rate: interpolation interval (default is 1Hz = 1 second intervals)
    :param overwrite: The outputfile, if exists, will be overwritten by setting to True
    :return:
    """
    try:
        # Create new netCDF file
        name_suffix = "_int_"+str(sampling_rate)+"_hz"
        dir_dst = os.path.join(dir_out, os.path.basename(os.path.dirname(file_in)) + name_suffix)
        os.makedirs(dir_dst, exist_ok=True)
        file_work = os.path.join(dir_work, os.path.basename(file_in) + name_suffix + '.nc')
        file_dst = os.path.join(dir_dst, os.path.basename(file_in) + name_suffix + '.nc')

        if os.path.exists(file_dst):
            print("File exists!", os.path.join(os.path.basename(file_in) + name_suffix + '.nc'))
            return True


        interval = 1.0/sampling_rate
        # define ecef and lla projections needed for conversion
        ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
        lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

        trans_latlon2ecef = pyproj.Transformer.from_proj(lla, ecef, always_xy=True)
        trans_ecef2latlon = pyproj.Transformer.from_proj(ecef, lla, always_xy=True)

        v = { 'tx_system': np.dtype(('U', 10)),
               'sample_time': np.int32,
               'sp_lon': np.float64,
               'sp_lat': np.float64,
               'rx_ant_gain_reflect': np.float64,
               'sp_crcg':  np.float64,
               'sp_rcg': np.float64,
               'sp_elevation_ang': np.float64,
               'track_id': np.int32,
               'tx_prn': np.int32}

        # copy  similar dictionaries to store intermediate results
        v1 = copy.deepcopy(v)
        v2 = copy.deepcopy(v)
        v3 = copy.deepcopy(v)
        # read all data
        nc_in = Dataset(file_in, 'r')
        for vname in v1.keys():
            v1[vname] = nc_in.variables[vname]
            # convert v3 variables to empty list
            v3[vname] = []

        for ut in set(v1['track_id'][:]):
            idx = np.where(v1['track_id'] == ut)
            # copy track data to new variable
            for vname in v1.keys():
                v2[vname] = v1[vname][idx]
            # sort data based on acquisition time
            sort_ind = np.argsort(v2['sample_time'])
            for vname in v2.keys():
                v2[vname] = v2[vname][sort_ind]

            # convert lat, lon, elevation to xyz
            # altitude information is not available form json files, therefore it is set as zero
            altitude = np.full(v2['sp_lon'].shape, 0.0, dtype=np.float64)
            x, y, z = trans_latlon2ecef.transform(v2['sp_lon'], v2['sp_lat'], altitude)

            for k in range(len(x) - 1):
                delta = v2['sample_time'][k + 1] - v2['sample_time'][k]
                if delta == 0:
                    continue
                int_num = int(round(delta / interval))

                xnew = np.linspace(x[k], x[k + 1], num=int_num, endpoint=False)
                ynew = np.linspace(y[k], y[k + 1], num=int_num, endpoint=False)
                znew = np.linspace(z[k], z[k + 1], num=int_num, endpoint=False)
                lon, lat, elv = trans_ecef2latlon.transform(xnew, ynew, znew)

                for vname in v3.keys():
                    if vname == 'sp_lon':
                        v3[vname].extend(lon)
                    elif vname == 'sp_lat':
                        v3[vname].extend(lat)
                    elif vname in ['tx_system', 'track_id', 'tx_prn']:
                        v3[vname].extend([v2[vname][k]]*int_num)
                    else:
                        v3[vname].extend(np.linspace(v2[vname][k], v2[vname][k + 1], num=int_num, endpoint=False))



        # write nc file
        nc = Dataset(file_work, 'w', diskless=True, persist=True)
        nc.createDimension('sample', len(v3['sp_lon']))
        for vname, variable in nc_in.variables.items():

            if vname == 'tx_system':
                x = nc.createVariable(vname, np.dtype(('U', 10)), 'sample')
                nc[vname][:] = np.array(v3[vname])
            else:
                x = nc.createVariable(vname, variable.datatype, 'sample')
                nc[vname].setncatts(nc_in[vname].__dict__)
                nc[vname][:] = np.ma.array(v3[vname])

        nc.close()
        compress_netcdf(file_work, file_dst)
    except OSError as e:
        print("Update error!..." + file_dst + str(e))
        return False

def _wrapper(dir_in, dir_out, dir_work, sampling_rate=2, mp_num=1, overwrite=False, node=None):

    files_in = glob.glob(os.path.join(dir_in, "sched_*.nc"))
    if node is not None:
        a = (node-1) * 70
        b = min([a+70, len(files_in)])
        files_in = files_in[a:b]

    if mp_num == 1:
        for file_in in files_in:
            interpolate_obs(file_in, dir_out, dir_work, sampling_rate = sampling_rate , overwrite=overwrite)
    else:
        prod_ftile = partial(interpolate_obs, dir_out=dir_out, dir_work=dir_work,
                             sampling_rate=sampling_rate, overwrite=overwrite)
        p = mp.Pool(processes=mp_num).map(prod_ftile, files_in)




def main():
    """
    This is wrapper program to read gnss-r coverage files in given directory, searches for the nearest equi7grid point to
    specular points and saves the indices as python object files in given output directory

    :param dir_in: Input directory of coverage files
    :param dir_out: Output directory to store indices as python object files
    :param interval: time interval (default is one second)
    :param num_pool: number of simultaneous processed for multiprocessing
    """
    node = 2
    dir_in = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-09-17_gnss-r_coverage_maps/schedules_att90_netcdf"
    dir_out = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-09-17_gnss-r_coverage_maps/schedules_att90_netcdf_int"
    dir_work = r"/home/ubuntu/_working_dir"
    _wrapper(dir_in, dir_out, dir_work, sampling_rate=10, mp_num=2, overwrite=False, node=node)

if __name__ == "__main__":
    main()




