import numpy as np
import os
from glob import glob
import pickle
import json
from datetime import datetime
from geopy.distance import geodesic
import h5py
from datetime import datetime
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile
from pygnssr.analytics.gnssr_analytics import cal_percentiles
from pygnssr.smap.SmapDataCube import get_l3_vars_template
import warnings
import multiprocessing as mp
from functools import partial
import dateutil.parser

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def open_smap_h5_files(dir_smap_l3, start_date=None, end_date=None):

    files = glob(os.path.join(dir_smap_l3, "*/*.h5"))

    # filter dates
    if (start_date is not None) and (end_date is not None):
        dates = np.array([datetime.strptime(os.path.basename(x)[-22:-14], "%Y%m%d") for x in files])
        ind = np.where((dates >= start_date) & (dates < end_date))
        files = list(np.array(files)[ind])
    h5_datasets = []
    for file in files:
        try:
            h5_datasets.append(h5py.File(file, mode='r'))
        except OSError as e:
            print("File reading error!..." + file + "\n" + str(e))

    return h5_datasets


def read_smap_sm_stack(h5_datasets=None, dir_smap_l3=None, sub_col=None, sub_row=None,
                       start_date=None, end_date=None, datetime_conversion=False):

    if (sub_col is not None) or (sub_row is not None):
        # TODO implment this
        print("subsetting is not implemented yet!!")
        return None

    if h5_datasets is None:
        h5_datasets = open_smap_h5_files(dir_smap_l3, start_date=start_date, end_date=end_date)

    # read time series of smap subset (image stacking)
    sm_am = '/Soil_Moisture_Retrieval_Data_AM/soil_moisture'
    sm_pm = '/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm'
    t_am = '/Soil_Moisture_Retrieval_Data_AM/tb_time_utc'
    t_pm = '/Soil_Moisture_Retrieval_Data_PM/tb_time_utc_pm'
    sm = []
    t = []
    for h5_ds in h5_datasets:
        sm.append(h5_ds[sm_am][:, :])
        sm.append(h5_ds[sm_pm][:, :])
        # resample smap time of acquisition (UTC)
        t.append(np.char.decode(h5_ds[t_am][:, :]))
        t.append(np.char.decode(h5_ds[t_pm][:, :]))

    _FillValue = h5_datasets[0][sm_am].attrs['_FillValue']
    valid_max = h5_datasets[0][sm_am].attrs['valid_max']
    valid_min = h5_datasets[0][sm_am].attrs['valid_min']
    invalid = np.logical_or(sm > valid_max, sm < valid_min)

    sm = np.ma.masked_where(sm == _FillValue, sm)
    sm = np.ma.masked_where(invalid, sm)
    t = np.ma.masked_where(sm.mask, t)

    if datetime_conversion:
        t = t.astype('O')
        #Converting time from string to datatime
        # vectorize string to date conversion
        vfunc = np.vectorize(_str_to_datetime)
        t[~t.mask] = vfunc(t[~t.mask])

    return t, sm

def read_ease_latlon_arr(dir_in, res=36000):
    """
    :param dir_in: directory of the ease grid lat/lon binary files
    :param res: EASE grid resolution in meters. Currently the supported resolutions are 9000 and 36000
    :return: SMAP longitudes and latitudes as 2D arrays
    """
    ease_dic = {'36000': {'lon': "EASE2_M36km.lons.964x406x1.double",
                          'lat': "EASE2_M36km.lats.964x406x1.double",
                          'row': 964, 'col': 406},
                '9000': {'lon': "EASE2_M09km.lons.3856x1624x1.double",
                         'lat': "EASE2_M09km.lats.3856x1624x1.double",
                         'row': 1624, 'col': 3856}}

    if not str(res) in ease_dic.keys():
        raise ValueError('The given grid resolution  ' + str(res) + '   is not supported!')

    lon_file = os.path.join(dir_in, ease_dic[str(res)]['lon'])
    lat_file = os.path.join(dir_in, ease_dic[str(res)]['lat'])

    lon_arr = np.fromfile(lon_file, dtype=np.double).reshape((ease_dic[str(res)]['row'], ease_dic[str(res)]['col']))[0, :]
    lat_arr = np.fromfile(lat_file, dtype=np.double).reshape((ease_dic[str(res)]['row'], ease_dic[str(res)]['col']))[:, 0]

    return lon_arr, lat_arr


def get_nearest_ease_grid_point(lon, lat, ease_lons=None, ease_lats=None, grid_res=36000, cal_dist=False):
    """
    :param lon: longitude of the selected point
    :param lat: latitude of the selected point
    :param ease_lons: (optional) longitudes used in the tiling (EASEGRID2)
                    e.g. in case of 36km grid, it is a 1-D array of 964 elements
    :param ease_lats: (optional) latitudes used in the tiling (EASEGRID2)
                    e.g. in case of 36km grid, it is a 1-D array of 406 elements
    :param grid_res: EASE grid resolution, default is 36000 meter
                    It is required if the ease_lons and ease_lats are not provided
    :param cal_dist: If True, the distance between the given point and the nearest EASE grid point will be calculated
    :return: row and column indices of the nearest EASE grid point
    """

    if ease_lons is None or ease_lats is None:
        ease_lons, ease_lats = read_ease_latlon_arr(dir_in=r"/home/ubuntu/datapool/external/ease_grid", res=grid_res)

    row = np.argmin(abs(lat-ease_lats))
    col = np.argmin(abs(lon-ease_lons))

    if cal_dist:
        pt = (ease_lats[row], ease_lons[col])
        dist = geodesic((lat, lon), pt).km
        print("Distance from the nearest EASE grid point "+ str(dist)+'km')

    return col, row


def _remove_older_files(files):
    """
    This programs removes older data files from the list if a new version of data file exists

    :param files: input h5 files
    :return:
    """
    # make a copy of input files list
    files_cleaned = files[:]
    files_nnn = np.array([x[-6:-3] for x in files])
    idx = np.where(files_nnn != '001')[0]
    files_2_remove=[]
    for i in idx:
        m = int(files_nnn[i]) - 1
        while m >= 1:
            files_2_remove.append(files[i][0:-6]+ str(m).zfill(3)+'.h5')
            m = m - 1

    for f in files_2_remove:
        if f in files_cleaned:
            files_cleaned.remove(f)
    return files_cleaned


def search_smap_h5_files(dir_smap_l3, start_date=None, end_date=None, return_h5_obj=False):

    files_all = glob(os.path.join(dir_smap_l3, "*/*.h5"))
    # remove duplicates (older data files that might exist with the same date/time and veriosn)
    files = _remove_older_files(files_all)
    # filter dates
    if (start_date is not None) and (end_date is not None):
        dates = np.array([datetime.strptime(os.path.basename(x)[-22:-14], "%Y%m%d") for x in files])
        ind = np.where((dates >= start_date) & (dates <= end_date))
        files = list(np.array(files)[ind])

    if return_h5_obj:
        h5_datasets = []
        for file in files:
            try:
                h5_datasets.append(h5py.File(file, mode='r'))
            except OSError as e:
                print("File reading error!..." + file + "\n" + str(e))
        return h5_datasets
    else:
        return files


def read_smap_sm_ts(h5_datasets=None, dir_smap_l3=None, col=None, row=None, lon=None, lat=None,
                    grid_res=36000, start_date=None, end_date=None, datetime_conversion=False):
    #todo needs revision
    if col is None or row is None:
        # get column and row arrays of smap (ease grid) corresponding to the target grid lat/lons
        col, row = get_nearest_ease_grid_point(lon, lat, grid_res=grid_res, cal_dist=False)

    if h5_datasets is None:
        h5_datasets = search_smap_h5_files(dir_smap_l3, start_date=start_date, end_date=end_date, return_h5_obj=True)

    # read time series of smap subset (image stacking)
    sm_am = '/Soil_Moisture_Retrieval_Data_AM/soil_moisture'
    sm_pm = '/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm'
    t_am = '/Soil_Moisture_Retrieval_Data_AM/tb_time_utc'
    t_pm = '/Soil_Moisture_Retrieval_Data_PM/tb_time_utc_pm'
    sm = []
    t = []
    for h5_ds in h5_datasets:
        sm.append(h5_ds[sm_am][row, col])
        sm.append(h5_ds[sm_pm][row, col])
        # resample smap time of acquisition (UTC)
        t.append(np.char.decode(h5_ds[t_am][row, col]))
        t.append(np.char.decode(h5_ds[t_pm][row, col]))

    _FillValue = h5_datasets[0][sm_am].attrs['_FillValue']
    valid_max = h5_datasets[0][sm_am].attrs['valid_max']
    valid_min = h5_datasets[0][sm_am].attrs['valid_min']
    invalid = np.logical_or(sm > valid_max, sm < valid_min)

    sm = np.ma.masked_where(sm == _FillValue, sm)
    sm = np.ma.masked_where(invalid, sm)
    t = np.ma.masked_where(sm.mask, t)

    if datetime_conversion:
        t = t.astype('O')
        #Converting time from string to datatime
        # vectorize string to date conversion
        vfunc = np.vectorize(_str_to_datetime)
        t[~t.mask] = vfunc(t[~t.mask])

    return t, sm


def _read_h5_vars_subset(h5_file, group_name, rmin, rmax, cmin, cmax, ridx, cidx):
    try:
        h5_ds = h5py.File(h5_file, mode='r')
        vars = get_l3_vars_template()
        for var_name in vars.keys():
            if var_name == 'processed_files':
                continue
            sub_var_am = h5_ds[group_name+'_AM/'+var_name][rmin:rmax+1, cmin:cmax+1]
            sub_var_pm = h5_ds[group_name+'_PM/'+var_name+'_pm'][rmin:rmax+1, cmin:cmax+1]
            vars[var_name] = [sub_var_am[ridx, cidx]]
            vars[var_name].append(sub_var_pm[ridx, cidx])
        print("SUCCESSFUL reading of " + os.path.basename(h5_file), datetime.now())
        h5_ds.close()
        return vars
    except Exception as e:
        print("FAILED reading of " + os.path.basename(h5_file), datetime.now())
        return None


def read_smap_subset(h5_files, col_arr, row_arr, time_conversion=False, mp_num=1):
    """
    This program reads smap h5 data subsets using the given column and row indices arrays

    :param h5_files: smap h5 data file paths
    :param col_arr: 2D array of column indices of the EASE grid
    :param row_arr: 2D array of row indices of the EASE grid
    :param time_conversion: convert observation time to datetime format
    :return: time and soil moisture variables
    """
    # get subset borders of SMAP data
    cmin = col_arr.min()
    cmax = col_arr.max()
    rmin = row_arr.min()
    rmax = row_arr.max()
    # column, row indices in subarray
    cidx = col_arr - cmin
    ridx = row_arr - rmin

    # hdf5 data parent name
    gname ='/Soil_Moisture_Retrieval_Data'

    """
    results=[]
    for f in h5_files:
        res = _read_h5_vars_subset(f, gname, rmin, rmax, cmin, cmax, ridx, cidx)
        print(res['processed_files'])
        stop = 0
        results.append(res)
    """
    if mp_num > 1:
        partial_func = partial(_read_h5_vars_subset, group_name=gname,
                               rmin=rmin, rmax=rmax, cmin=cmin, cmax=cmax, ridx=ridx, cidx=cidx)
        results = mp.Pool(processes=mp_num).map(partial_func, h5_files)
    else:
        results = []
        for f in h5_files:
            results.append(_read_h5_vars_subset(f, gname, rmin, rmax, cmin, cmax, ridx, cidx))

    # get the smap variables list
    smap_vars = get_l3_vars_template()
    init = True
    successful_read = False
    flist = []
    for k, h5_file in enumerate(h5_files):
        fname = os.path.basename(h5_file)
        if results[k] is None:
            continue
        flist.append(fname)
        successful_read = True
        for var_name in smap_vars.keys():
            if init:
                smap_vars[var_name] = results[k][var_name]
            else:
                if var_name != 'processed_files':
                    smap_vars[var_name].append(results[k][var_name][0])
                    smap_vars[var_name].append(results[k][var_name][1])
        init = False
    if not successful_read:
        return None
    smap_vars['processed_files'] = np.array(flist)

    sample_date = smap_vars['processed_files'][0].split('_')[5]
    sample_file = os.path.join(os.path.dirname(os.path.dirname(h5_files[0])),
                               sample_date[0:4]+'.'+sample_date[4:6]+'.'+sample_date[6:8],
                               smap_vars['processed_files'][0])
    # read attributes of soil moisture variable
    h5_ds_sample = h5py.File(sample_file, mode='r')
    # get soil moisture fill value and valid range
    _FillValue = h5_ds_sample[gname+'_AM/soil_moisture'].attrs['_FillValue']
    valid_max = h5_ds_sample[gname+'_AM/soil_moisture'].attrs['valid_max']
    valid_min = h5_ds_sample[gname+'_AM/soil_moisture'].attrs['valid_min']
    #invalid = np.logical_or(smap_vars['soil_moisture'] > valid_max, smap_vars['soil_moisture'] < valid_min)
    h5_ds_sample.close()
    # soil moisture emaks
    mask = smap_vars['soil_moisture'] == _FillValue
    # convert numpy arrays to mask arrays
    for var_name in smap_vars.keys():
        if var_name == 'processed_files':
            smap_vars[var_name] = np.array(smap_vars[var_name])
            continue
        smap_vars[var_name] = np.ma.masked_where(mask, smap_vars[var_name])
        # todo Should be the variables masked where the soil moisture is outside the valid range?
        # smap_vars[var_name] = np.ma.masked_where(invalid, smap_vars[var_name])
        # convert 'tb_time_utc' from byte to string
        if var_name == 'tb_time_utc':
            smap_vars[var_name] = smap_vars[var_name].astype(str)

    # set soil moisture fill value
    np.ma.set_fill_value(smap_vars['soil_moisture'], _FillValue)

    if time_conversion:
        # Converting time from string to datatime
        tt = np.ma.empty(smap_vars['tb_time_utc'].shape, dtype=datetime)
        tt.mask = mask
        if len(smap_vars['tb_time_utc'][~mask]) > 0:
            vfunc = np.vectorize(_str_to_datetime)
            tt[~mask] = vfunc(smap_vars['tb_time_utc'][~mask])
        smap_vars['tb_time_utc'] = tt
    return smap_vars


def resample_smap(lon_arr, lat_arr, ease_grid_res, dir_smap_l3=None, h5_files=None, start_date=None, end_date=None):
    """
    :param lon_arr:
    :param lat_arr:
    :param start_date: ignored if h5_files are provided
    :param end_date: ignored if h5_files are provided
    :param dir_smap_l3: ignored if h5_files are provided
    :param h5_files: SMAP h5 data file paths
    :return:
    """

    # TODO add other resampling methods as well (not just nearest neighbourhood)
    # TODO producing ease grid lat lons are repeating for each tile!

    ease_lons, ease_lats = read_ease_latlon_arr(dir_in=r"/home/ubuntu/datapool/external/ease_grid", res=ease_grid_res)

    # get column and row arrays of smap (ease grid) corresponding to the target grid lat/lons
    vfunc = np.vectorize(get_nearest_ease_grid_point, excluded=["ease_lats", "ease_lons"])
    col_arr, row_arr = vfunc(lon_arr, lat_arr, ease_lats=ease_lats, ease_lons=ease_lons)

    # TODO: remove this part to _wrapper. It is repeating for ech equi7tile!
    if h5_files is None:
        h5_files = search_smap_h5_files(dir_smap_l3, start_date=start_date, end_date=end_date)

    t, sm = read_smap_subset(h5_files, col_arr, row_arr)

    return t, sm


def _str_to_datetime(time):

    """
    try:
        t = datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception as e:
        t = None
    return t
    """
    # todo check why sometimes SMAP hour is 24???
    time = time.replace('T24', 'T00')
    t = datetime.fromisoformat(time[:-5])

    return t


def _wrapper_cal_smap_percentiles(start_date,  end_date, ease_grid_res, dir_smap_l3=None, dir_out=None,
                                  grid_res=3000, sgrids=None, num_pool=1, overwrite=False):

    # open SMAP data files
    h5_files = search_smap_h5_files(dir_smap_l3, start_date=start_date, end_date=end_date)

    grid = Equi7Grid(grid_res)
    if sgrids is None:
        sgrids = grid._static_sgrid_ids
    else:
        if not isinstance(sgrids, list):
            sgrids = [sgrids]

    for sgrid in sgrids:
        dir_tile_list = r"/home/ubuntu/datapool/internal/misc/land_tile_list"
        tile_list_file = os.path.join(dir_tile_list, sgrid.upper()+"_T6_LAND_EQUTORIAL.json")
        with open(tile_list_file, "r") as f:
            tnames = json.load(f)
        ftiles =[sgrid.upper()+str(grid.res)+"M_"+tname for tname in tnames]

        for ftile in ftiles:
            # write smap percentiles as python objects
            dir_sub = os.path.join(dir_out, ftile[0:7])
            os.makedirs(dir_sub, exist_ok=True)
            file_out = os.path.join(dir_sub, 'SMAP_PERCENTILES_'+ftile+'.pkl')
            if os.path.exists(file_out) and not overwrite:
                warnings.warn(ftile+'  exists! Set overwrite keyword as True to replace it')

            tile = Equi7Tile(ftile)
            tsize = tile.shape[0]
            y_idx_arr = np.tile(np.array(range(tsize)), (tsize, 1))
            x_idx_arr = np.tile(np.array(range(tsize)), (tsize, 1)).T

            x_arr = x_idx_arr*tile.res + tile.llx + tile.res/2.0
            y_arr = y_idx_arr*tile.res + tile.lly + tile.res/2.0

            lon_arr, lat_arr = grid.equi7xy2lonlat(sgrid, x_arr, y_arr)

            _, sm_smap = resample_smap(lon_arr, lat_arr, ease_grid_res, h5_files=h5_files)
            # plt.imshow(sm_smap[10, :, :]) ; plt.show()

            perc_smap = np.full((10, tsize, tsize), np.nan, dtype=np.float32)
            for i in range(tile.shape[0]):
                for j in range(tile.shape[0]):
                    perc_smap[:, i, j] = cal_percentiles(sm_smap, i, j, nbins=10)

            with open(file_out, 'wb') as f:
                pickle.dump(perc_smap, f)

def main():

    print('Nothing to do!')
    #todo move following lines to a separate function for processing smap data and calculate percentiles
    """
    dir_dpool = r"/home/ubuntu/datapool"
    #dir_smap_l3 = os.path.join(dir_dpool, "external", "SMAP", "SPL3SMP.006")
    dir_smap_l3 = os.path.join(dir_dpool, "external", "SMAP", "SPL3SMP_E.003")
    if os.path.basename(dir_smap_l3) == "SPL3SMP.006":
        ease_grid_res = 36000
    elif os.path.basename(dir_smap_l3) == "SPL3SMP_E.003":
        ease_grid_res = 9000
    else:
        raise ValueError("SMAP product is unknown!")

    dir_out = os.path.join(dir_dpool, "internal", "datacube", "smap", os.path.basename(dir_smap_l3), "percentiles")

    start_date = datetime(2017, 4, 1)
    end_date = datetime(2020, 4, 1)

    sgrids = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']
    for sg in sgrids:
        stime_fsgrid = datetime.now()
        _wrapper_cal_smap_percentiles(start_date,  end_date, ease_grid_res, dir_smap_l3=dir_smap_l3, dir_out=dir_out,
                                      grid_res=3000, sgrids=sg, num_pool=7, overwrite=False)

        print("Total processing time for " + sg+": "+str(datetime.now()-stime_fsgrid))

        #print("The job is deactivated! Check the python code!")
        print(datetime.now(), "SMAP calculations finished!")
    """

if __name__ == "__main__":
    main()




