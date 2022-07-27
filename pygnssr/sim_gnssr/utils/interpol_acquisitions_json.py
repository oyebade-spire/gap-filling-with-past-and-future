import json
import numpy as np
import os
import glob
from multiprocessing import Pool
from datetime import datetime, timedelta
from itertools import repeat
import pyproj


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"

def interpolate_obs(file_in, file_out, sampling_rate =1, overwrite=False):
    """
    This program retrieves the Specular Point (SP) locations from GNSS-R coverage file, searches for
    corresponding equi7grid point and stores the indices as Python object in given output directory

    :param file_in: GNSS-R json data file
    :param file_out: full path of output file
    :param sampling_rate: interpolation interval (default is 1Hz = 1 second intervals)
    :param overwrite: The outputfile, if exists, will be overwritten by setting to True
    :return:
    """
    interval = 1.0/sampling_rate

    #define ecef and lla projections needed for conversion
    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')


    trans_latlon2ecef= pyproj.Transformer.from_proj(lla, ecef, always_xy=True)
    trans_ecef2latlon = pyproj.Transformer.from_proj(ecef, lla, always_xy=True)

    jtracks = json.load(open(file_in))
    for i, jtrack in enumerate(jtracks):
        loctime = []; latitude = []; longitude = []; elevation = []; antgain = []; crcg = []; rcg = []
        for obs in jtrack['command']['rayGeolocation']:
            loctime.append(datetime.strptime(obs['loctime'], '%Y-%m-%dT%H:%M:%SZ'))
            latitude.append(obs['latitude'])
            longitude.append(obs['longitude'])
            elevation.append(obs['elevation'])
            antgain.append(obs['antgain'])
            crcg.append(obs['crcg'])
            rcg.append(obs['rcg'])

        # convert lat, lon, elevation to xyz
        x, y, z = trans_latlon2ecef.transform(longitude, latitude, elevation)

        loctimenew = []; xnew = []; ynew = []; znew = []; antgainnew = []; crcgnew = []; rcgnew = []
        for k in range(len(x) - 1):
            delta = loctime[k + 1] - loctime[k]
            step = timedelta(seconds=interval)
            int_num = round(delta.total_seconds() / interval)
            tnew = [loctime[k]+step*i for i in range(int_num)]
            loctimenew.extend([t.strftime('%Y-%m-%dT%H:%M:%S.%fZ') for t in tnew])
            xnew.extend(np.linspace(x[k], x[k + 1], num=int_num, endpoint=False))
            ynew.extend(np.linspace(y[k], y[k + 1], num=int_num, endpoint=False))
            znew.extend(np.linspace(z[k], z[k + 1], num=int_num, endpoint=False))
            antgainnew.extend(np.linspace(antgain[k], antgain[k + 1], num=int_num, endpoint=False))
            crcgnew.extend(np.linspace(crcg[k], crcg[k + 1], num=int_num, endpoint=False))
            rcgnew.extend(np.linspace(rcg[k], rcg[k + 1], num=int_num, endpoint=False))

        # append the last measurement
        xnew.append(x[-1]); ynew.append(y[-1]);  znew.append(z[-1])
        loctimenew.append(loctime[-1].strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
        antgainnew.append(antgain[-1]); crcgnew.append(crcg[-1]);  rcgnew.append(rcg[-1])

        # convert x,y,z back to lat, lon, elevation
        lon, lat, elv = trans_ecef2latlon.transform(xnew, ynew, znew)
        newobj = []
        for i in range(len(xnew)):
            dict= {"loctime": loctimenew[i],
                   "longitude": lon[i],
                   "latitude": lat[i],
                   "elevation": elv[i],
                   "antgain": antgainnew[i],
                   "crcg": crcgnew[i],
                   "rcg": rcgnew[i]
                   }
            newobj.append(dict)
        jtrack['command']['rayGeolocation'] = newobj

    with open(file_out, 'w') as f:
        json.dump(jtracks, f, indent="  ")

def _wrapper(dir_in, dir_out, sampling_rate=2, num_pool=1):

    files_in = glob.glob(os.path.join(dir_in, "*.json"))

    name_suffix = "_int_"+str(sampling_rate)+"_hz"
    dir_out_int = os.path.join(dir_out, os.path.basename(dir_in) + name_suffix)
    os.makedirs(dir_out_int, exist_ok=True)
    files_out = [os.path.join(dir_out_int, os.path.basename(x) + name_suffix + '.json') for x in files_in]

    # create a pool of simultaneous processes
    p = Pool(num_pool)
    results = p.starmap(interpolate_obs, list(zip(files_in, files_out, repeat(sampling_rate))))

def main():
    """
    This is wrapper program to read gnss-r coverage files in given directory, searches for the nearest equi7grid point to
    specular points and saves the indices as python object files in given output directory

    :param dir_in: Input directory of coverage files
    :param dir_out: Output directory to store indices as python object files
    :param interval: time interval (default is one second)
    :param num_pool: number of simultaneous processed for multiprocessing
    """

    dir_in = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-09-17_gnss-r_coverage_maps/schedules"
    dir_out = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-09-17_gnss-r_coverage_maps/schedule_int"

    _wrapper(dir_in, dir_out, sampling_rate=2, num_pool=7)

if __name__ == "__main__":
    main()




