import numpy as np

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from netCDF4 import Dataset, date2num, num2date
from datetime import datetime

from pygnssr.common.math.ts_modulation import calibrate, scale, temporal_mean
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile
from pygnssr.spire_gnssr.Spire_gnssrDataCube import Spire_gnssrDataCube


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"



def cal_acqui_time(nc_time, units, calendar):
    # mask bad values (for some reason it is not already masked!!!)
    nc_time = np.ma.masked_greater_equal(nc_time, 1e+30, copy=False)
    # create a new masked array to hold time in datetime format
    acq_time = np.ma.empty(nc_time.shape, dtype=datetime)
    acq_time.mask = nc_time.mask
    # concert time from float number to datetime format
    # todo for some reason the following remarked line is not working!!!!
    ### acq_time[~nc_time.mask] = [np.datetime64(num2date(tt, units, calendar=calendar)).astype(datetime)
    ###                           for tt in nc_time[~nc_time.mask]]
    acq_time[~nc_time.mask] = [num2date(tt, units, calendar=calendar) for tt in nc_time[~nc_time.mask]]

    return acq_time


def read_comb_ssm_l2u1_ssm(lon, lat, dir_comb_ssm_l2u1, grid_res,  agg_pix):

    ftile, x_index, y_index = get_e7idx(grid_res, lon, lat)

    nc = get_dcube(ftile, 'comb-ssm', dir_comb_ssm_l2u1, data_level='l2u1')

    # start indices
    si = x_index - (x_index % agg_pix)
    sj = y_index - (y_index % agg_pix)
    # end indices
    ei = si + agg_pix
    ej = sj + agg_pix

    if nc is not None:
        rssm = nc['cygnss']['rssm'][:, si:ei, sj:ej]
        rssm = rssm.flatten()

        cssm = nc['cygnss']['cssm'][:, si:ei, sj:ej]
        cssm = cssm.flatten()

        sp_lon = nc['cygnss']['sp_lon'][:, si:ei, sj:ej]
        sp_lon = sp_lon.flatten()

        sp_lat = nc['cygnss']['sp_lat'][:, si:ei, sj:ej]
        sp_lat = sp_lat.flatten()

        # calculate cygnss time
        units = nc['cygnss']['time_utc'].units
        calendar = nc['cygnss']['time_utc'].calendar
        cygn_t = cal_acqui_time(nc['cygnss']['time_utc'][:, si:ei, sj:ej].data.flatten(), units, calendar)

        mask = np.ma.getmaskarray(cygn_t)
        sort_ind = cygn_t[~mask].data.argsort()
        rssm[~mask] = rssm[~mask][sort_ind]
        cssm[~mask] = cssm[~mask][sort_ind]
        sp_lon[~mask] = sp_lon[~mask][sort_ind]
        sp_lat[~mask] = sp_lat[~mask][sort_ind]
        cygn_t[~mask] = cygn_t[~mask][sort_ind]
        #------------------------------------------------------
        smap_sm = nc['smap']['sm'][:, si:ei, sj:ej]
        smap_sm = smap_sm.flatten()

        # calculate smap time
        units = nc['smap']['time_utc'].units
        calendar = nc['smap']['time_utc'].calendar
        smap_t = cal_acqui_time(nc['smap']['time_utc'][:, si:ei, sj:ej].data.flatten(), units, calendar)

        mask = np.ma.getmaskarray(smap_t)
        sort_ind = smap_t[~mask].data.argsort()
        smap_sm[~mask] = smap_sm[~mask][sort_ind]
        smap_t[~mask] = smap_t[~mask][sort_ind]
        #------------------------------------------------------

        return cygn_t, rssm, cssm, sp_lon, sp_lat, smap_t, smap_sm
    else:
        print('netCDF file path not valid!')


def read_cygnss_l2_ssm(lon, lat, dir_cygnss_l2, grid_res,  agg_pix):

    ftile, x_index, y_index = get_e7idx(grid_res, lon, lat)
    nc = get_cygnss_l2_dcube(ftile, dir_cygnss_l2)

    # start indices
    si = x_index - (x_index % agg_pix)
    sj = y_index - (y_index % agg_pix)
    # end indices
    ei = si + agg_pix
    ej = sj + agg_pix

    if nc is not None:
        rssm = nc.variables['rssm'][:, si:ei, sj:ej]
        rssm = rssm.flatten()

        cssm = nc.variables['cssm'][:, si:ei, sj:ej]
        cssm = cssm.flatten()

        inc = nc.variables['sp_inc_angle'][:, si:ei, sj:ej]
        inc = inc.flatten()

        sp_lon = nc.variables['sp_lon'][:, si:ei, sj:ej]
        sp_lon = sp_lon.flatten()

        sp_lat = nc.variables['sp_lat'][:, si:ei, sj:ej]
        sp_lat = sp_lat.flatten()

        qflags = nc.variables['quality_flags'][:, si:ei, sj:ej]
        qflags = qflags.flatten()

        # calculate time
        units = nc.variables['ddm_timestamp_utc'].units
        calendar = nc.variables['ddm_timestamp_utc'].calendar
        t = cal_acqui_time(nc.variables['ddm_timestamp_utc'][:, si:ei, sj:ej].data.flatten(), units, calendar)


        sort_ind = t[~t.mask].data.argsort()
        rssm[~t.mask] = rssm[~t.mask][sort_ind]
        cssm[~t.mask] = cssm[~t.mask][sort_ind]
        inc[~t.mask] = inc[~t.mask][sort_ind]
        sp_lon[~t.mask] = sp_lon[~t.mask][sort_ind]
        sp_lat[~t.mask] = sp_lat[~t.mask][sort_ind]
        qflags[~t.mask] = qflags[~t.mask][sort_ind]
        t[~t.mask] = t[~t.mask][sort_ind]

        return t, rssm, cssm, inc, sp_lon, sp_lat, qflags
    else:
        print('netCDF file path not valid!')


def read_cygnss_l1_data(lon, lat, dir_cygnss_l1, grid_res,  agg_pix):

    ftile, x_index, y_index = get_e7idx(grid_res, lon, lat)
    nc = get_cygnss_l1_dcube(ftile, dir_cygnss_l1)

    # start indices
    si = x_index - (x_index % agg_pix)
    sj = y_index - (y_index % agg_pix)
    # end indices
    ei = si + agg_pix
    ej = sj + agg_pix

    if nc is not None:
        rx_gain = nc.variables['sp_rx_gain'][:, si:ei, sj:ej]
        rx_gain = rx_gain.flatten()

        qflags = nc.variables['quality_flags'][:, si:ei, sj:ej]
        qflags = qflags.flatten()

        # calculate time
        units = nc.variables['ddm_timestamp_utc'].units
        calendar = nc.variables['ddm_timestamp_utc'].calendar
        t = cal_acqui_time(nc.variables['ddm_timestamp_utc'][:, si:ei, sj:ej].data.flatten(), units, calendar)

        sort_ind = t[~t.mask].data.argsort()
        rx_gain[~t.mask] = rx_gain[~t.mask][sort_ind]
        qflags[~t.mask] = qflags[~t.mask][sort_ind]
        t[~t.mask] = t[~t.mask][sort_ind]

        return t, rx_gain, qflags
    else:
        print('netCDF file path not valid!')


def get_cygnss_l2_params(dir_cygnss_l2, res, lon, lat, agg_pix):
    ftile, x_index, y_index = get_e7idx(res, lon, lat)
    print(ftile)
    grid_res = float(ftile.split('_')[0][2:-1]) / 1000.0
    nc = get_cygnss_l2_dcube(ftile, dir_cygnss_l2)

    # start indices
    si = x_index - (x_index % agg_pix)
    sj = y_index - (y_index % agg_pix)
    # end indices
    ei = si + agg_pix
    ej = sj + agg_pix

    if nc is not None:
        dry = nc.variables['dry'][si:ei, sj:ej]
        dry = dry.flatten()

        wet = nc.variables['wet'][si:ei, sj:ej]
        wet = wet.flatten()

        inc_slope = nc.variables['inc_slope'][si:ei, sj:ej]
        inc_slope = inc_slope.flatten()

        rfl = nc.variables['rfl'][:, si:ei, sj:ej]
        rfl = rfl.flatten()

        nrfl = nc.variables['nrfl'][:, si:ei, sj:ej]
        nrfl = nrfl.flatten()

        rssm = nc.variables['rssm'][:, si:ei, sj:ej]
        rssm = rssm.flatten()

        qflags = nc.variables['quality_flags'][:, si:ei, sj:ej]
        qflags = qflags.flatten()

        # calculate time
        units = nc.variables['ddm_timestamp_utc'].units
        calendar = nc.variables['ddm_timestamp_utc'].calendar
        t = cal_acqui_time(nc.variables['ddm_timestamp_utc'][:, si:ei, sj:ej].data.flatten(), units, calendar)

        sort_ind = t[~t.mask].data.argsort()
        t[~t.mask] = t[~t.mask][sort_ind]
        rfl[~t.mask] = rfl[~t.mask][sort_ind]
        nrfl[~t.mask] = nrfl[~t.mask][sort_ind]
        rssm[~t.mask] = rssm[~t.mask][sort_ind]

        return grid_res, t[~t.mask], rfl[~t.mask], nrfl[~t.mask], rssm[~t.mask], qflags[
            ~t.mask], dry.mean(), wet.mean(), inc_slope.mean()
    else:
        print('netCDF file path not valid!')




def read_spire_gnssr_l1_data(lon, lat, dir_spire_gnssr_l1, grid_res,  agg_pix):

    ftile, x_index, y_index = get_e7idx(grid_res, lon, lat)
    sub_dir_in = os.path.join(dir_spire_gnssr_l1, ftile.split('_')[0])
    dc = Spire_gnssrDataCube(ftile, 'L1', sub_dir_in)
    dc.read()

    # start indices
    si = x_index - (x_index % agg_pix)
    sj = y_index - (y_index % agg_pix)
    # end indices
    ei = si + agg_pix
    ej = sj + agg_pix

    if dc.nc is not None:
        snr = dc.nc.variables['reflect_snr_at_sp'][:, si:ei, sj:ej]
        snr = snr.flatten()

        rfl = 10.0 * np.log10(dc.nc.variables['reflectivity_at_sp'][:, si:ei, sj:ej])
        rfl = rfl.flatten()

        qflags = dc.nc.variables['quality_flags'][:, si:ei, sj:ej]
        qflags = qflags.flatten()

        t= dc.nc.variables['sample_time'][:, si:ei, sj:ej]
        t=t.flatten()
        """
        # calculate time
        units = dc.nc.variables['sample_time'].units
        calendar = dc.nc.variables['sample_time'].calendar
        t = cal_acqui_time(dc.nc.variables['sample_time'][:, si:ei, sj:ej].data.flatten(), units, calendar)

        sort_ind = t[~t.mask].data.argsort()
        snr[~t.mask] = snr[~t.mask][sort_ind]
        rfl[~t.mask] = rfl[~t.mask][sort_ind]
        qflags[~t.mask] = qflags[~t.mask][sort_ind]
        t[~t.mask] = t[~t.mask][sort_ind]
        """

        dc.close_nc()
        return t, snr, rfl, qflags
    else:
        print('netCDF file path not valid!')










def get_e7idx(grid_res, lon, lat):
    ftile, _, _, x_index, y_index = Equi7Grid(grid_res)._lonlat2equi7xy_idx(lon,lat)
    return ftile, x_index, y_index

def get_dcube(ftile, dcube_name, dir_dcube, data_level=1):
    level = "L"+str(data_level)
    dc_file = os.path.join(dir_dcube, dcube_name.lower(), "dataset", level,
                           ftile.split('_')[0], dcube_name.upper() + "_" + level + "_" +ftile+".nc")
    if os.path.exists(dc_file):
        result = Dataset(dc_file, 'r')
    else:
        result = None
    return result


def dcube_downsample(arr, agg_pix, operation=None):
    '''
    This code is used to downscample a tree dimensional array (data cube in from of [t, x, y])
    (downsampling is meant here just re-indexing)

    or a two dimensional parameters in form of masked array of [x, y]
    (in this case paramter will be averaged)


    :param dcube_arr:
    :param agg_pix: number of grid points to be aggregated
    :param operation: (optional) if provided, then the operation will be applied on aggregated
                        measurement instead of just indexing.operations: 'mean', 'max', 'min'
    :return: downsampled array with same content of input array but reshaped/re-indexed
    '''

    if agg_pix == 1:
        return arr

    if len(arr.shape) == 3:
        if (arr.shape[1] % agg_pix) != 0 or (arr.shape[1] != arr.shape[2]):
            raise ValueError('n should be a submultiple of array size')
    elif len(arr.shape) == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError('Size of x and y dimensions are not equal!')
    else:
        raise ValueError('Input array should have either two or three dimensions')

    xysize = arr.shape[1]
    xysize_out = int(xysize / agg_pix)

    # create indices needed to create output grid pixels
    y_indices = np.tile(range(0, xysize, agg_pix), (1, xysize_out)).reshape(xysize_out, xysize_out)
    x_indices = y_indices.transpose()

    if operation is not None:
        if not operation.lower() in ['mean', 'min', 'max']:
            raise ValueError('Given operation is not supported!')
        ds_arr = np.ma.empty((xysize_out, xysize_out), dtype=arr.dtype)
    else:
        tsize = arr.shape[0]
        tsize_out = tsize * agg_pix * agg_pix
        ds_arr = np.ma.empty((tsize_out, xysize_out, xysize_out), dtype=arr.dtype)

    ds_arr.mask = True
    for xi, yi in zip(x_indices.flatten(), y_indices.flatten()):

        # start indices
        si = xi - (xi % agg_pix)
        sj = yi - (yi % agg_pix)
        # end indices
        ei = si + agg_pix
        ej = sj + agg_pix

        if operation is not None:
            ds_arr[int(si/agg_pix), int(sj/agg_pix)] = getattr(arr[si:ei, sj:ej], operation)()
        else:
            ds_arr[:, int(si/agg_pix), int(sj/agg_pix)] = arr[:, si:ei, sj:ej].reshape(tsize_out)

    return ds_arr
