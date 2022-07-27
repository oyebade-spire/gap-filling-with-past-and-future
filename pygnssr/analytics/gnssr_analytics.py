import numpy as np
from pygnssr.common.math.outlier_removal import outlier_removal
from pygnssr.cygnss.cygnss_utils import get_mask, get_sv_bias
from scipy import stats, interpolate
import warnings as warn


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


def cal_rfl(nc, bias_corr=False, ppa=False):
    # todo: the function should be generalized (currently it is working with cygnss datacube)
    if nc is not None:
        if ppa:
            ddm_peak = nc.variables['ddm_peak'][:, :, :]
            ddm_noise = nc.variables['ddm_noise'][:, :, :]
            snr = 10.0 * np.log10((ddm_peak / ddm_noise) - 1.0)
        else:
            snr = nc.variables['ddm_snr'][:, :, :]
            snr = 10.0 * np.log10(10 ** (snr / 10.0) - 1.0)

        rgain = nc.variables['sp_rx_gain'][:, :, :]
        rx = nc.variables['rx_to_sp_range'][:, :, :].astype('float32')
        tx = nc.variables['tx_to_sp_range'][:, :, :].astype('float32')

        # calculate Range-Corrected Gain (RCG)
        crl = 10.0 * np.log10((rx + tx) ** 2)
        crcg = rgain - crl + 160  # 160 db correction

        # calculate RFL
        ptgt = 10.0 * np.log10(nc.variables['gps_eirp'][:, :, :])
        const_1 = -14.424927980943421 # 10.0 * np.log10(0.19 ** 2)
        const_2 = 21.984197280441926 #10.0 * np.log10((4 * np.pi) ** 2)
        rfl = snr - ptgt - crcg + const_2 - const_1

        # apply bias correction
        if bias_corr:
            vfunc = np.vectorize(get_sv_bias)
            bias = vfunc(nc.variables['sv_num'][:, :, :])
            rfl =rfl + bias

        #apply mask
        cygnss_mask = get_mask(nc.variables['quality_flags'])
        rfl = np.ma.masked_where(cygnss_mask, rfl)

        # todo: check if filtering SNR helps to improve ssm quality
        # rfl = np.ma.masked_where(snr < -2.3, rfl)


        return rfl


def cal_slope(inc, rfl, x_index, y_index):

    rfl = np.ma.masked_invalid(rfl)
    inc = np.ma.masked_invalid(inc)
    inc = np.ma.masked_greater_equal(inc, 65)

    vfunc = np.vectorize(_cal_slope, excluded=['inc', 'rfl'])
    slope, intcp = vfunc(inc=inc, rfl=rfl, x_index=x_index, y_index=y_index)
    return slope, intcp


def _cal_slope(inc, rfl, x_index, y_index):
    # todo: the function should be generalized (currently it is working equi7  tile indices)
    """

    :param inc:
    :param rfl:
    :param x_index:
    :param y_index:
    :return:
    """
    try:
        x = inc[:, x_index, y_index].flatten()
        y = rfl[:, x_index, y_index].flatten()

        y = outlier_removal(y, degree=0.5)

        slope, intercept, r_value, p_value, std_err = stats.linregress(x[~y.mask], y[~y.mask])

        # alternative function for linear regression
        #poly_coef = np.polyfit(x[~y.mask], y[~y.mask], 1)
        #slope = poly_coef[0]
        #intercept = poly_coef[1]

    except Exception as e:
        #print(e)
        slope = np.nan
        intercept = np.nan

    return slope, intercept


def cal_nrfl(inc, rfl, slope, ref_inc=40):
    """
    calculate normalized Reflectivity
perc_smap
    :param rfl:
    :param inc:
    :param slope:
    :param ref_inc:
    :return: normalized Reflectivity
    """
    # todo:check if masking high incidence angles help to improve ssm quality
    #inc = np.ma.masked_greater_equal(inc, 60)

    # replace NaN in slope parameter with 0 (this will result --> nrfl=rfl, no normalization applied)
    slope = np.nan_to_num(slope)


    nrfl = rfl + slope * (ref_inc - inc)
    if np.ma.isMaskedArray(rfl):
        nrfl = np.ma.masked_where(rfl.mask, nrfl)
    if np.ma.isMaskedArray(inc):
        nrfl = np.ma.masked_where(inc.mask, nrfl)

    return nrfl


def cal_drywet(nrfl, x_indices, y_indices):
    nrfl = np.ma.masked_invalid(nrfl)
    vfunc = np.vectorize(_cal_drywet, excluded=['nrfl'])
    dry, wet = vfunc(nrfl=nrfl, x_index=x_indices, y_index=y_indices)
    return dry, wet


def _cal_drywet(nrfl, x_index, y_index):
    # todo: the function should be generalized (currently it is working equi7  tile indices)
    # convert input to numpy masked array and mask invalid

    arr = nrfl[:, x_index, y_index].flatten()
    try:
        # calculate 5th and 95th percentiles
        p10 = np.percentile(arr[~arr.mask], 10)
        p90 = np.percentile(arr[~arr.mask], 90)

        const = 10.0 * (p90 - p10) / (90.0 - 10.0)

        dry = p10 - const
        wet = p90 + const
    except Exception as e:
        dry = np.nan
        wet = np.nan

    return dry, wet


def cal_rssm(nrfl, dry, wet):

    rssm = (nrfl - dry)*100.0/(wet - dry)
    # convert to masked array
    rssm = np.ma.MaskedArray(rssm)


    # apply corrections
    pflag = np.zeros(rssm.shape, dtype=np.uint8)

    #idx = (rssm < 0) & (rssm >= -10)
    #rssm[idx] = 0
    #pflag[idx] = 1

    #idx = (rssm > 100) & (rssm <= 110)
    #rssm[idx] = 100
    #pflag[idx] = 2

    rssm = np.ma.masked_outside(rssm, 0, 100)
    # todo change rssm data type to int8

    return rssm, pflag


def cal_cssm(src, perc_src, perc_ref, x_indices, y_indices):

    src = np.ma.masked_invalid(src)
    perc_src = np.ma.masked_invalid(perc_src)
    perc_ref = np.ma.masked_invalid(perc_ref)
    # TODO: for unknown reason vectorizing CSSM is not working consistently
    #vfunc = np.vectorize(_cal_cssm, excluded=['src', 'perc_src', 'perc_ref'])
    #cssm = vfunc(src=src, perc_src=perc_src, perc_ref=perc_ref, x_index=x_indices, y_index=y_indices)
    # here is the loop solution
    cssm = np.full_like(src, np.nan)
    for i in range(x_indices.shape[0]):
        for j in range(x_indices.shape[1]):
            cssm[:, i, j] = _cal_cssm(src, perc_src, perc_ref, x_indices[i, j], y_indices[i, j])

    return cssm


def _cal_cssm(src, perc_src, perc_ref, x_index, y_index):
    # initialize the scaled variable with source array
    var = src[:, x_index, y_index].flatten()
    src_scl = np.ma.masked_all_like(var)
    psrc = perc_src[:, x_index, y_index].flatten()
    pref = perc_ref[:, x_index, y_index].flatten()
    if not (np.any(psrc.mask) or np.any(pref.mask)):
        try:
            src_scl[~var.mask] = gen_cdf_match(var[~var.mask], psrc, pref, k=1)
        except Exception as e:
            src_scl[:] = np.nan
            src_scl = np.ma.masked_invalid(src_scl)
            return src_scl

    return src_scl


def cal_percentiles(src, x_index, y_index, q=None, nbins=100):

    """
    """
    # convert input to numpy masked array and mask invalid
    arr = np.ma.masked_invalid(src[:, x_index, y_index])
    arr = arr.compressed()
    try:
        if q is None:
            q = np.linspace(0, 100, nbins)

        percentiles = np.array(np.percentile(arr, q))
        percentiles = unique_percentiles_interpolate(percentiles, percentiles=q)
    except Exception as e:
        percentiles = np.full(len(q), np.nan)
        return percentiles

    return percentiles


    """
    def cal_percentiles(src, x_index, y_index, q=None):
    """
    """
    :param src: source time series
    :param q: percentile to compute, which must be between 0 and 100 inclusive.
    :return:
    """
    """
    # convert input to numpy masked array and mask invalid
    arr = np.ma.masked_invalid(src[:, x_index, y_index])
    arr = arr.compressed()
    try:
        if q is not None:
            percentile = np.percentile(arr, q)
            #perc_src = unique_percentiles_interpolate(arr, percentiles=percentiles)

        else:
            percentile = np.nan
    except Exception as e:
        percentile = np.nan
        return percentile

    return percentile

    """


def gen_cdf_match(src,
                  perc_src, perc_ref,
                  min_val=None, max_val=None,
                  k=1):
    #TODO: revising
    """
    General cdf matching:

    1. computes discrete cumulative density functions of
       src- and ref at the given percentiles
    2. computes continuous CDFs by k-th order spline fitting
    3. CDF of src is matched to CDF of ref

    Parameters
    ----------
    src: numpy.array
        input dataset which will be scaled
    perc_src: numpy.array
        percentiles of src
    perc_ref: numpy.array
        percentiles of reference data
        estimated through method of choice, must be same size as
        perc_src
    min_val: float, optional
        Minimum allowed value, output data is capped at this value
    max_val: float, optional
        Maximum allowed value, output data is capped at this value
    k : int, optional
        Order of spline to fit

    Returns
    -------
    CDF matched values: numpy.array
        dataset src with CDF as ref
    """
    # InterpolatedUnivariateSpline uses extrapolation
    # outside of boundaries so all values can be rescaled
    # This is important if the stored percentiles were generated
    # using a subset of the data and the new data has values outside
    # of this original range

    #NaN handling: If the input arrays contain nan values,
    # the result is not useful, since the underlying spline fitting routines
    # cannot deal with nan . A workaround is to use zero weights for not-a-number data points:
    try:
        inter = interpolate.InterpolatedUnivariateSpline(perc_src, perc_ref, k=k)
    except Exception:
        # here we must catch all exceptions since scipy does not raise a proper
        # Exception
        warn("Too few percentiles for chosen k.")
        return np.full_like(src, np.nan)

    scaled = inter(src)
    if max_val is not None:
        scaled[scaled > max_val] = max_val
    if min_val is not None:
        scaled[scaled < min_val] = min_val

    return scaled


def unique_percentiles_interpolate(perc_values, percentiles=[0, 5, 10, 30, 50, 70, 90, 95, 100], k=1):

    #TODO: revising needed
    """
    Try to ensure that percentile values are unique
    and have values for the given percentiles.

    If only all the values in perc_values are the same.
    The array is unchanged.

    Parameters
    ----------
    perc_values: list or numpy.ndarray
        calculated values for the given percentiles
    percentiles: list or numpy.ndarray
        Percentiles to use for CDF matching
    k: int
        Degree of spline interpolation to use for
        filling duplicate percentile values

    Returns
    -------
    uniq_perc_values: numpy.ndarray
        Unique percentile values generated through linear
        interpolation over removed duplicate percentile values
    """
    uniq_ind = np.unique(perc_values, return_index=True)[1]
    if len(uniq_ind) == 1:
        uniq_ind = np.repeat(uniq_ind, 2)
    uniq_ind[-1] = len(percentiles) - 1
    uniq_perc_values = perc_values[uniq_ind]

    inter = interpolate.InterpolatedUnivariateSpline(np.array(percentiles)[uniq_ind],
                                                     uniq_perc_values, k=k, ext=0, check_finite=True)
    uniq_perc_values = inter(percentiles)
    return uniq_perc_values
