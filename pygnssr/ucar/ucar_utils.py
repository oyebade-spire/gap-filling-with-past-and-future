import numpy as np
import os
from glob import glob
from datetime import datetime
from geopy.distance import geodesic
from netCDF4 import Dataset, date2num, num2date
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gzip

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def read_ucar_latlon_arr(ucar_sample_file):
    """
    :return: SMAP longitudes and latitudes as 2D arrays
    """
    dsin = Dataset(ucar_sample_file, 'r')
    lons_ucar = dsin.variables['longitude']
    lats_ucar = dsin.variables['latitude']

    return lons_ucar, lats_ucar


def get_ucar_nearest_grid_point(lon, lat, lons_ucar, lats_ucar):
    """
    :param lon: longitude of the selected point
    :param lat: latitude of the selected point
    :param lons_ucar: 2D array of UCAR longitudes
    :param lats_ucar: 2D array of UCAR latitudes
    :return:
    """

    lon_min = np.min(lons_ucar[:, :])
    lon_max = np.max(lons_ucar[:, :])
    lat_min = np.min(lats_ucar[:, :])
    lat_max = np.max(lats_ucar[:, :])
    lon_stp = (lon_max - lon_min)/ (lons_ucar.shape[0]-1)
    lat_stp = (lat_max - lat_min)/ (lats_ucar.shape[1]-1)

    col_fast = int((lon - lon_min)/lon_stp)
    row_fast = int((lat_max - lat)/lat_stp)
    # define a 40*40 subset pixels around the approximated point
    a = max(0, col_fast-20)
    b = min(lons_ucar.shape[0], col_fast+21)
    c = max(0, row_fast-20)
    d = min(lons_ucar.shape[1], row_fast+21)

    # calculate distance of the given point from the points within the search area and find the nearest point
    dist=np.full((lons_ucar.shape[0], lons_ucar.shape[1]), 9999 )
    for i in range(b-a+1):
        for j in range(d-c+1):
            x= a+i
            y= c+j
            pt = (lats_ucar[0, y], lons_ucar[x, 0])
            dist[x, y] = geodesic((lat, lon), pt).km

    min_dist = np.min(dist)
    print("Distance from the nearest UCAR grid point "+ str(np.min(dist))+'km')
    if min_dist > 50:
        print('Selected point for UCAR is out of observation range!')
        return None, None
    else:
        ind = np.unravel_index(np.argmin(dist, axis=None), lons_ucar.shape)
        col = ind[0]
        row = ind[1]
        return col, row



def read_ucar_sm(dir_ucar_l3, col, row, start_date=None, end_date=None):
    files = glob(os.path.join(dir_ucar_l3, "*/*.nc"))

    dates = []
    for f in files:
        year = int(os.path.basename(f).split('_')[-2])
        doy = int(os.path.basename(f).split('_')[-1][0:3])
        dates.append(datetime(year, 1, 1) + timedelta(doy - 1))

    # filter dates
    if (start_date is not None) and (end_date is not None):
        ind = np.where((dates >= start_date) & (dates <= end_date))
        files = list(np.array(files)[ind])
        dates = dates[ind]

    sm = []
    for f in files:
        dsin = Dataset(f, 'r')
        sm.append(dsin.variables['SM_daily'][col, row])
        dsin.close()

    return np.array(dates), np.ma.array(sm)




def main(ucar_dir, ucar_sample_file, lon, lat):

    lons_ucar, lats_ucar = read_ucar_latlon_arr(ucar_sample_file)
    col, row = get_ucar_nearest_grid_point(lon, lat, lons_ucar, lats_ucar)
    print(lons_ucar[col, row], lats_ucar[col, row])

    print(lons_ucar.shape)

    """
    time, sm = read_ucar_sm(ucar_dir, col, row)
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.set_ylim(0, 0.7)
    ax1.plot(time, sm, '-or', markersize=4)
    plt.show() 
    """



if __name__ == "__main__":

    ucar_dir = r"D:\Vahid\03_datapool\external\cygnss_sm_ucar\level3"
    ucar_sample_file = os.path.join(ucar_dir, r"2019\ucar_cu_cygnss_sm_v1_2019_001.nc")
    lon = -99.87
    lat = 33.45

    main(ucar_dir, ucar_sample_file, lon, lat)





