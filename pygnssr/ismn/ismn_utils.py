import numpy as np
import os
from glob import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import geodesic


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def read_ismn_latlon_arr(dir_ismn_latlon):
    """
    :param dir_ismn_latlon: directory of the ismn staions lon/lat list
    :return: longitudes and latitudes as panda dataframe
    """
    # read location information of all networks and stations
    files = glob(dir_ismn_latlon + "/*")
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(file))
    df_latlon = pd.concat(dfs, axis=0, ignore_index=True)

    return df_latlon



def get_ismn(lon, lat, dir_ismn, df_latlon, search_radius=12):
    dist = []
    for i in range(len(df_latlon)):
        pt = (df_latlon["LATITUDES"][i], df_latlon["LONGITUDES"][i])
        dist.append(geodesic((lat, lon), pt).km)

    min_dist = min(dist)

    # return is the distance is less than 12km
    if min_dist <= search_radius:
        idx = dist.index(min_dist)

        pattern = os.path.join(dir_ismn, df_latlon["NETWORK"][idx], df_latlon["DIR_NAME"][idx],  "*.stm")
        files = glob(pattern)

        var = []
        dfrom = []
        dto = []
        for f in files:
            fname = os.path.basename(f).split("_")
            var.append(fname[3])
            dfrom.append(float(fname[4]))
            dto.append(float(fname[5]))

        # filter files to top layer soil moisture
        ind = np.where((np.array(var) == 'sm') & (np.array(dto) <= 0.40))
        if len(ind[0]) != 0:
            file = files[ind[0][0]]

            df_ismn = pd.read_fwf(file, header=None)
            df_ismn.columns = ["utc_ndate", "utc_ntime", "utc_adate", "utc_atime", "cse", "network", "station", "lat",
                               "lon", "elevation", "depth_from", "depth_to", "var",
                               "qflag1", "qflag2"]

            # calculate datatime
            t_ismn = [datetime.strptime(x.strip() + y.strip(), '%Y/%m/%d%H:%M') for x, y in
                      zip(df_ismn["utc_adate"], df_ismn["utc_atime"])]

            return t_ismn, df_ismn, idx

    return None, None, None



def main(dir_ismn_latlon, dir_ismn, lon, lat):

    df_latlon = read_ismn_latlon_arr(dir_ismn_latlon)
    t_ismn, df_ismn, df_latlon_idx = get_ismn(lon, lat, dir_ismn, df_latlon, search_radius=12)
    sm_ismn = df_ismn['var'].to_numpy()

    print(df_latlon_idx)
    print(df_latlon["LONGITUDES"][df_latlon_idx])
    print(df_latlon["LATITUDES"][df_latlon_idx])

    """
    fig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.set_ylim(0, 0.7)
    ax1.plot(t_ismn, sm_ismn, '-or', markersize=4)
    plt.show() 
    """


if __name__ == "__main__":
    dir_ismn_latlon = r"D:\Vahid\03_datapool\external\ismn\_location_files"
    dir_ismn = r"D:\Vahid\03_datapool\external\ismn\networks"

    lon = -99.87
    lat = 33.45

    main(dir_ismn_latlon, dir_ismn, lon, lat)




