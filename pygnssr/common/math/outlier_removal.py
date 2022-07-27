import numpy as np

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"

def outlier_removal(x, degree=1.5, side=2):
    """
    :param x: Input data distribution (
    :param degree: constant number to be multiplied by Inter-quartile range (IQR)
                    default is 1.5 for removing moderate outliers.
                    use e.g. degree = 3 to remove only extreme outliers
    :param side: an option to remove outliers from one side or both sides of distribution
                    0: low values
                    1: high values
                    2: (default) both low and high values

    :return: a numpy masked array. outliers are masked
    """
    # convert input to numpy masked array and mask invalid
    a = np.ma.masked_invalid(x)

    # calculate 25th and 75th percentiles
    p25 = np.percentile(a[~a.mask], 25)
    p75 = np.percentile(a[~a.mask], 75)

    # calculate inter-quartile range
    iqr = p75 - p25

    # mask outlier outside inter-quartile range multiplied by constant
    if side == 0:
        # mask outlier from low value measurements
        a = np.ma.masked_less(a, p25 - iqr * degree)
    if side == 1:
        # mask outlier from high value measurements
        a = np.ma.masked_greater(a, p25 + iqr * degree)
    else:
        # mask outlier from both sides
        a = np.ma.masked_outside(a, p25-iqr*degree, p75+iqr*degree)

    return a






