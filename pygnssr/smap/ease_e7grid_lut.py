import os
import pickle
import numpy as np
import json
from datetime import datetime
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile
from pygnssr.smap.smap_utils import read_ease_latlon_arr, get_nearest_ease_grid_point
from functools import partial
import multiprocessing as mp
import warnings

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def _cal_ease_lut(lon_arr, lat_arr, ease_grid_res=None, ease_grid_dir=None, ease_lons=None, ease_lats=None):
    """
    This program searches for corresponding equi7grid point and stores the indices as Python object
    """
    if ease_lons is None or ease_lats is None:
        if ease_grid_dir is None:
            raise ValueError('EASE grid data directory should be provided!')
        ease_lons, ease_lats = read_ease_latlon_arr(dir_in=ease_grid_dir, res=ease_grid_res)

    # get column and row arrays of smap (ease grid) corresponding to the target grid lat/lons
    vfunc = np.vectorize(get_nearest_ease_grid_point, excluded=["ease_lats", "ease_lons"])
    col_lut, row_lut = vfunc(lon_arr, lat_arr, ease_lats=ease_lats, ease_lons=ease_lons)

    return col_lut, row_lut


def _gen_ease_e7_lut(ftile, ease_lons, ease_lats, dir_out, ease_grid_res=None, overwrite=False):
    """
    Generates a Look-Up-Table (LUT) between EASE grid and Equi7grid
    :param ftile: Equi7 Full Tile name
    :param ease_lons:  One dimensional array of EASE grid longitudes
    :param ease_lats: One dimensional array of EASE grid latitudes
    :param ease_grid_res: EASE2.0 grid resolution (36000 and 9000 meters are supported)
    :return:
    """

    # write smap percentiles as python objects
    dir_sub = os.path.join(dir_out, ftile[0:7])
    os.makedirs(dir_sub, exist_ok=True)
    file_out = os.path.join(dir_sub, 'ease_equi7_lut_'+ftile+'.pkl')
    if os.path.exists(file_out) and not overwrite:
        warnings.warn(ftile+'  exists! Set overwrite keyword as True to replace it')

    tile = Equi7Tile(ftile)
    tsize = tile.shape[0]
    y_idx_arr = np.tile(np.array(range(tsize)), (tsize, 1))
    x_idx_arr = np.tile(np.array(range(tsize)), (tsize, 1)).T

    x_arr = x_idx_arr*tile.res + tile.llx + tile.res/2.0
    y_arr = y_idx_arr*tile.res + tile.lly + tile.res/2.0

    grid = Equi7Grid(tile.res)
    lon_arr, lat_arr = grid.equi7xy2lonlat(ftile.split("_")[0][0:2], x_arr, y_arr)

    col_lut, row_lut = _cal_ease_lut(lon_arr, lat_arr, ease_grid_res=ease_grid_res,
                                     ease_lons=ease_lons, ease_lats=ease_lats)

    with open(file_out, 'wb') as f:
        pickle.dump((col_lut, row_lut), f)


def _wrapper(ftiles, ease_lons, ease_lats, dir_out, num_pool=1, overwrite=False):

    # loop solution -------------------------------------------------------------------------------------
    for ftile in ftiles:
        _gen_ease_e7_lut(ftile, ease_lons, ease_lats, dir_out, overwrite=overwrite)

    # loop solution -------------------------------------------------------------------------------------
    #prod_ftile = partial(_gen_ease_e7_lut, dir_work=dir_work,
    #                     dir_l1=dir_l1, dir_smap_perc=dir_smap_perc, agg_pix=agg_pix, gen_l2_flag=gen_l2_flag,
    #                     dir_l2=dir_l2, dir_l2p=dir_l2p, overwrite=overwrite)
    #p = mp.Pool(processes=num_pool).map(prod_ftile, ftiles)


def main():
    print(datetime.now(), " LUT creation started from python code ...")

    # determine input/outöut grid resolutions
    ease_grid_res = 9000
    e7_grid_res = 3000

    dir_dpool = r"/home/ubuntu/datapool"
    ease_grid_dir = os.path.join(dir_dpool, "external", "ease_grid")
    # todo include ease_grid_res in file names and put results in distinct output directory
    dir_out = os.path.join(dir_dpool, "internal", "misc", "ease_equi7_lut")

    ease_lons, ease_lats = read_ease_latlon_arr(dir_in=ease_grid_dir, res=ease_grid_res)

    grid = Equi7Grid(e7_grid_res)
    sgrids = ['AS', 'NA', 'SA', 'AF', 'EU', 'OC']

    ftiles = []
    for sgrid in sgrids:
        dir_tile_list = os.path.join(dir_dpool, "internal", "misc", "land_tile_list")
        tile_list_file = os.path.join(dir_tile_list, sgrid.upper() + "_T6_LAND.json")
        with open(tile_list_file, "r") as f:
            tile_names = json.load(f)
        ftiles.extend([sgrid.upper()+str(grid.res)+"M_"+tile for tile in tile_names])

    _wrapper(ftiles, ease_lons, ease_lats, dir_out, num_pool=7, overwrite=True)

    print(datetime.now(), "Data indexing/LUT creation is finished!")

if __name__ == "__main__":
    main()






