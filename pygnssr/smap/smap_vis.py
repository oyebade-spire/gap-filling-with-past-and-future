import numpy as np
import os
from glob import glob
import pickle
import json
from datetime import datetime
from geopy.distance import geodesic
import h5py
from ease_grid import EASE2_grid
from datetime import datetime
from pygnssr.common.time.get_time_intervals import get_time_intervals
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile
from pygnssr.analytics.gnssr_analytics import cal_percentiles
from pygnssr.smap.smap_utils import read_smap_sm_stack
from pygnssr.common.utils.gdalport import write_tiff
from osgeo import osr


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"



def _wrapper_geotif(date_int, dir_smap_l3, dir_out):
    for st, et in zip(date_int[0], date_int[1]):
        # check if start and end time are list (in case of climatologic has been set as True)
        if type(st) is list:
            print("NOT IMPLEMENTED YET!")
        else:
            t_arr, smap_arr = read_smap_sm_stack(dir_smap_l3=dir_smap_l3, start_date=st, end_date=et)
            dst_arr = smap_arr.mean(axis=0)

            fname = st.strftime("%Y%m%d") + "_" + et.strftime("%Y%m%d") + "_SMAP_L3SM"  ".tif"
            _write_dst(dst_arr, fname, dir_out)



def _write_dst(arr, fname, dir_out):
    # set NaN value
    nan_val = -9999
    arr = arr.filled(nan_val)

    # metadata
    # -------------------------------------------
    geotags={}
    geotags['no_data_val'] = nan_val
    geotags['description'] = "SMAP Level-3 soil moisture product (36km)"
    geotags['metadata'] = {'creator': "SPIRE GLOBAL",
                           'processing_date':datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           'source': "SMAP Level-3 surface soil moisture product ",
                           'release': "test_v0.5",
                           'grid': 'EASE Grid'}


    src = osr.SpatialReference()
    src.ImportFromEPSG(6933)
    geotags['spatialreference'] = src.ExportToPrettyWkt()
    geotags['geotransform'] = (-17367530.44516138, 36000, 0, 7314540.79258289, 0, -36000.0)


    os.makedirs(dir_out, exist_ok=True)
    dst_file = os.path.join(dir_out, fname)
    write_tiff(dst_file, arr, tiff_tags=geotags)
    ########################################################################

def _smap2geotiff(file_in, dir_out):

    h5_ds = h5py.File(file_in, mode='r')

    # read time series of smap subset (image stacking)


    sm_am = h5_ds['/Soil_Moisture_Retrieval_Data_AM/soil_moisture'][:, :]
    sm_pm = h5_ds['/Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm'][:, :]
    _FillValue = h5_ds['/Soil_Moisture_Retrieval_Data_AM/soil_moisture'].attrs['_FillValue']

    sm_am = np.ma.masked_where(sm_am == _FillValue, sm_am)
    sm_pm = np.ma.masked_where(sm_pm == _FillValue, sm_pm)

    _write_dst(sm_am, os.path.basename(file_in)+"_am.tif", dir_out)
    _write_dst(sm_pm, os.path.basename(file_in)+"_pm.tif", dir_out)



def main():
    file_in = r"/home/ubuntu/datapool/external/smap/SPL3SMP_E/2020.12.04/SMAP_L3_SM_P_E_20201204_R17000_002.h5"
    dir_work = r"/home/ubuntu/_working_dir"
    _smap2geotiff(file_in, dir_work)


    """
    dir_external = r"/home/ubuntu/datapool/external"
    dir_internal = r"/home/ubuntu/datapool/internal"
    dir_smap_l3 = os.path.join(dir_external, "smap", "SPL3SMP_E")
    dir_work = r"/home/ubuntu/_working_dir"
    #dir_out = os.path.join(dir_work, "2020-05-09_sm_analysis", "_SMAP_monthly")
    dir_out = os.path.join(dir_work, "2020-05-09_sm_analysis", "_SMAP_daily")

    start_date = datetime(2020, 12, 1)
    end_date = datetime(2020, 12, 4)

    date_int = get_time_intervals(start_date, end_date, interval_type='daily')

    _wrapper_geotif(date_int, dir_smap_l3, dir_out)
    """
    print('done!')

if __name__ == "__main__":
    main()




