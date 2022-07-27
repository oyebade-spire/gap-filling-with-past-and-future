import numpy as np
from pygnssr.common.math.outlier_removal import outlier_removal
import pandas as pd
from copy import deepcopy as cpy
from datetime import datetime
#from pytesmo.scaling import lin_cdf_match

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


def calibrate(ts_in, ts_ref):
    """
    :param ts_in: time series of uncalibrated variable in form of numpy masked array
    :param ts_ref: time series of reference variable in form of numpy masked array
    :return: calibrated variable
    """
    calib_ts = cpy(ts_in)

    # check if elements are invalid
    val_idx = np.isfinite(ts_in) & np.isfinite(ts_ref) & ~ts_in.mask & ~ts_ref.mask

    x = ts_in[val_idx] - ts_in[val_idx].mean()
    y = ts_ref[val_idx] - ts_ref[val_idx].mean()

    y = outlier_removal(y, degree=3)

    poly_coef = np.polyfit(x[~y.mask], y[~y.mask], 1)
    calib_slope = poly_coef[0]
    calib_intcp = poly_coef[1]

    calib_ts[val_idx] = calib_slope * x + ts_ref[val_idx].mean()

    return calib_ts

def cdf_match(ts_in, ts_ref):
    """
    :param ts_in:
    :param ts_ref:
    :return:
    """
    pass




def scale(ts_in, ts_ref):
    """
    :param ts_in: time series of input variable in form of numpy masked array
    :param ts_ref: time series of reference variable in form of numpy masked array
    :return: scaled time series
    """
    # scales x time series according to y time series
    x_min = ts_in[~ts_in.mask].min()
    x_max = ts_in[~ts_in.mask].max()
    y_min = np.min(ts_ref)
    y_max = np.max(ts_ref)

    ts_scaled = ((ts_in - x_min) * (y_max - y_min) / (x_max - x_min)) + y_min
    return ts_scaled


def moving_mean(ts, num_fore=0, num_aft=0):
    ts_out = np.zeros_like(ts, dtype=np.float64)
    dim_len = ts.shape[0]
    for i in range(dim_len):
        a = max(0, i - num_aft)
        b = min(dim_len, i + num_fore + 1)
        ts_out[i] = np.mean(ts[a:b])
    return ts_out


def temporal_mean(ts_time, ts_var, comp_type, ts_name='variable'):
    ttt = [np.datetime64(tt).astype(datetime) for tt in ts_time]
    df = pd.DataFrame({ts_name: ts_var}, index=ttt)


    if comp_type is not None:
        result = df.resample(comp_type).mean()
        """
           if comp_type.lower() == 'daily':
               result = df.resample('D').mean()
           elif comp_type.lower() == 'weekly':
               result = df.resample('W').mean()
           elif comp_type.lower() == 'monthly':
               result = df.resample('M').mean()
        """
    else:
        result = df
    return result

    #todo following line is obsolete, tobe removed after harmonization
    #return result.index.to_pydatetime(), result['var'].to_numpy()
