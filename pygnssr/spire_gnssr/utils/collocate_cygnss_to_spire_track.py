import os
import click
import json
from datetime import datetime
import numpy as np
from pygnssr.cygnss.CygnssDataCube import CygnssDataCube
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile
import multiprocessing as mp
from functools import partial
from netCDF4 import Dataset
import pickle
import glob
from collections import OrderedDict
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from pygnssr.analytics.gnssr_analytics import cal_rfl

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"

def _get_indices(file_idx, ftile=None):
    with open(file_idx, "r") as f:
        # loading look-up-table between SMAP EASE grid and Equi7 grid
        dic = json.load(f)
    ft_arr = np.array(dic['ftile'])
    ix_arr = np.array(dic['ix'])
    iy_arr = np.array(dic['iy'])
    if ftile is not None:
        # indices matching the e7grid tile name
        ind = ft_arr == ftile
        ft_arr = ft_arr[ind]
        # sub set array of location indices
        ix_arr = ix_arr[ind]
        iy_arr = iy_arr[ind]
    return ft_arr, ix_arr, iy_arr

def _get_template():
    v1 = {'e7_lon': np.float32,
          'e7_lat': np.float32}

    v2 = {'ddm_timestamp_utc': np.float64,
          #'spacecraft_num': np.int8,
          'ddm_snr': np.float32,
          'prn_code': np.int8,
          #'sv_num': np.int32,
          'sp_inc_angle': np.float32,
          'sp_rx_gain': np.float32,
          #'gps_eirp': np.float32,
          #'gps_tx_power_db_w': np.float32,
          #'gps_ant_gain_db_i': np.float32,
          #'ddm_noise_floor': np.float32,
          #'rx_to_sp_range': np.int32,
          #'tx_to_sp_range': np.int32,
          #'ddm_nbrcs': np.float32,
          #'ddm_les': np.float32,
          #'nbrcs_scatter_area': np.float32,
          #'les_scatter_area': np.float32,
          #'ddm_brcs_uncert': np.float32,
          'quality_flags': np.int32,
          # todo: remark following variables if v2.1 is data is used
          'brcs_sp': np.float32,
          'brcs_peak': np.float32}

    v3 = { 'rfl': np.float32 }


    return v1, v2, v3

def _create_nc_out(file_path, dir_cygnss_v3, smaple_count=None):
    sample_file = os.path.join(dir_cygnss_v3, "AF3000M", "CYGNSS_L1_AF3000M_E090N078T6.nc")
    nc_sample = Dataset(sample_file, 'r')
    # output nc
    nc = Dataset(file_path, 'w', diskless=True, persist=True)

    nc.history = 'Created on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')
    nc.creator = 'SPIRE GLOBAL'
    nc.description = 'Historical CYGNSS Level-1 Data resampled to Spire Level-1 track'
    nc.source = 'Extracted from CYGNSS Level-1 Data Cube'
    nc.version = '0.2'
    nc.createDimension('sample', smaple_count)
    nc.createDimension('ts', None)

    v1, v2, v3 = _get_template()

    for p, q in v1.items():
        v1[p] = nc.createVariable(p, q, ('sample'), fill_value=-9999.0)
        if p == 'e7_lon':
            nc[p].setncatts(OrderedDict({'long_name': 'equi7 grid point longitude',
                                         'units': 'degrees',
                                         'comment': 'longitude of the equi7 grid point.'}))
        elif p == 'e7_lat':
            nc[p].setncatts(OrderedDict({'long_name': 'equi7 grid point latitude',
                                         'units': 'degrees',
                                         'comment': 'latitude of the equi7 grid point.'}))

    for p, q in v2.items():
        v2[p] = nc.createVariable(p, q, ('sample', 'ts'))
        nc[p].setncatts(nc_sample[p].__dict__)

    for p, q in v3.items():
        v3[p] = nc.createVariable(p, q, ('sample', 'ts'), fill_value=-9999.0)
        if p == 'rfl':
            nc[p].setncatts(OrderedDict({'long_name': 'reflectivity',
                                         'units': 'dB',
                                         'comment': 'Reflectivity calculated from CYGNSS version 2.1 data using '
                                                    'the radar equation for coherent measurements'}))


    return nc

def _calculate_e7_lonlat(ftile, nc):
    # calculate equi7grid lat/lon
    e7tile = Equi7Tile(ftile)
    e7grid = Equi7Grid(e7tile.res)
    size = nc.dimensions['x'].size
    y_idx_arr = np.tile(np.array(range(size)), (size, 1))
    x_idx_arr = np.tile(np.array(range(size)), (size, 1)).T
    x_arr = x_idx_arr*e7tile.res + e7tile.llx + e7tile.res/2.0
    y_arr = y_idx_arr*e7tile.res + e7tile.lly + e7tile.res/2.0
    lon_arr, lat_arr = e7grid.equi7xy2lonlat(ftile[0:2], x_arr, y_arr)

    return lon_arr, lat_arr

def _get_cache_vars(sample_size, x_size, y_size):
    v1, v2, v3 = _get_template()

    names_list = list(v2.keys())
    names_list.extend(list(v3.keys()))

    formats_list = list(v2.values())
    formats_list.extend(list(v3.values()))

    tp = np.dtype({'names': names_list, 'formats': formats_list})
    cache_arr = np.ma.masked_all((sample_size, x_size, y_size), dtype=tp)
    return cache_arr

def _gen_cygnss_track(file_idx, dir_cygnss, dir_work, dir_out, overwrite=False):

    file_org = os.path.basename(file_idx).split('__')[0]
    file_temp = os.path.join(dir_work, 'cygnss-collocated_'+os.path.basename(file_org))
    file_dst = os.path.join(dir_out, file_org[26:36], 'cygnss-collocated_'+os.path.basename(file_org))

    if os.path.exists(file_dst) and not overwrite:
        print('File exists! set overwrite keyword!')
        return False

    ft_arr, ix_arr, iy_arr= _get_indices(file_idx)

    # initialize output netcdf file
    nc_out = _create_nc_out(file_temp, dir_cygnss, smaple_count=len(ft_arr))
    v1, v2, v3 = _get_template()

    ft_old = "null"
    for i, (ft, x, y) in enumerate(zip(ft_arr, ix_arr, iy_arr)):
        if ft != ft_old:
            try:
                # read cygnss datacube
                dc_l1 = CygnssDataCube(ft, "L1", os.path.join(dir_cygnss,  ft.split('_')[0]), flag='r')
                cache_arr = _get_cache_vars(dc_l1.nc.dimensions['sample'].size,
                                               dc_l1.nc.dimensions['x'].size, dc_l1.nc.dimensions['y'].size)

                # lon_arr, lat_arr = _calculate_e7_lonlat(ft, dc.nc)
                for vname in v2.keys():
                    cache_arr[vname][:, :, :] = dc_l1.nc[vname][:, :, :]

                cache_arr['rfl'][:, :, :] = cal_rfl(dc_l1.nc, bias_corr=False)

                dc_l1.close_nc()

            except Exception as e:
                print(str(e))
                dc_l1 = None
                cache_arr = None
            ft_old = ft
        if dc_l1 is not None:
            try:
                # nc_out.variables['e7_lon'][i] = lon_arr[x, y]
                # nc_out.variables['e7_lat'][i] = lat_arr[x, y]
                for vname in v2.keys():
                    nc_out.variables[vname][i, :] = cache_arr[vname][:, x, y]
                for vname in v3.keys():
                    nc_out.variables[vname][i, :] = cache_arr[vname][:, x, y]
            except Exception as e:
                print(str(e))

        else:
            # nc_out.variables[vname][i, 0] = nan_val
            print('nc is None')
    nc_out.close()

    # compress the netcdf file and move to destination directory
    os.makedirs(os.path.dirname(file_dst), exist_ok=True)
    compress_netcdf(file_temp, file_dst)

def _wrapper(files_idx, dir_work, dir_cygnss, dir_out, mp_num=1, overwrite=False):

    if mp_num == 1:
        for file_idx in files_idx:
            _gen_cygnss_track(file_idx, dir_cygnss, dir_work, dir_out, overwrite=overwrite)
    else:
        prod_ftile = partial(_gen_cygnss_track, dir_cygnss=dir_cygnss,
                             dir_work=dir_work, dir_out=dir_out, overwrite=overwrite)
        p = mp.Pool(processes=mp_num).map(prod_ftile, files_idx)

def main():
    """
    This program
    - reads the spire_gnssr level-1 index file, retrieves desired cygnss variables using the information in index file,
    applies requested operation and stores results as python object in given output directory

    """
    log_start_time = datetime.now()
    print(log_start_time, " Data resampling started from python code ...")
    # ----------------------------------------------------------
    dir_work = r"/home/ubuntu/_working_dir"
    dir_dpool = r"/home/ubuntu/datapool"
    dir_prod = os.path.join(dir_dpool, "internal", "temp_working_dir",
                            "2020-12-01_spire_cygnss_collocated_tracks", "prod_0.3.7")

    #################    resampling CYGNSS version 3   ###########################
    dir_out = os.path.join(dir_prod, "data_v3")
    # TODO: change _get_template if resampling v2.1
    #dir_cygnss = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L1")
    dir_cygnss = os.path.join(dir_dpool, "internal", "temp_working_dir", "2020-12-09_cygnss_v3",
                              "internal", "datacube", "cygnss", "dataset", "L1")
    dir_spire_idx = os.path.join(dir_dpool, "internal", "datacube", "spire_gnssr", "prod-0.3.7",
                                 "spire_gnssr_e7_indices")
    ##############################################################################
    mp_num = 8 # number of multiple processing
    overwrite = False
    # ----------------------------------------------------------
    #######################         temp
    dir_search = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-12-01_spire_cygnss_collocated_tracks/prod_v0.2.2/selected_plots"
    sel_plots = glob.glob(os.path.join(dir_search, "*/*.png"))
    files_idx = []
    for sel_plot in sel_plots:
        fname_plot = os.path.basename(sel_plot)
        fname = fname_plot.replace('_v0.2.2.nc', '_v0.3.7.nc')
        fname_idx = 'Spire'+fname.split('_Spire')[1][0:-4]+'__e7indices_3000m.json'
        files_idx.append(os.path.join(dir_spire_idx, fname_idx[26:36], fname_idx))

    #######################         temp


    #files_idx = glob.glob(os.path.join(dir_spire_idx, "*/*.json"))

    _wrapper(files_idx, dir_work, dir_cygnss, dir_out, mp_num=mp_num, overwrite=overwrite)



    print(datetime.now(), "Data resampling is finished!")
    print("Total processing time "+str(datetime.now()-log_start_time))

if __name__ == "__main__":
    main()
