from datetime import datetime, timedelta


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"

def date_to_doys(date=None , num_days=1):
    """
    :param date: date in datetime format, default is current date
    :param num_days: number of days before the given date
    :return: a list of year and day-of-year according to the given number of days
    """
    if date is None:
        date = datetime.now()
    year_list = [(date - timedelta(x)).year for x in range(num_days)]
    doy_list = [(date - timedelta(x)).timetuple().tm_yday for x in range(num_days)]

    return year_list, doy_list


def date_to_ymd(date=None, num_days=1):
    """
    :param date: date in datetime format, default is current date
    :param num_days: number of days before the given date
    :return: a list of year, month, and day according to the given number of days
    """
    if date is None:
        date = datetime.now()
    year_list = [(date - timedelta(x)).year for x in range(num_days)]
    month_list = [(date - timedelta(x)).month for x in range(num_days)]
    day_list = [(date - timedelta(x)).day for x in range(num_days)]

    return year_list, month_list, day_list


def date_to_ymd_str(date=None, num_days=1):
    """
    :param date: date in datetime format, default is current date
    :param num_days: number of days before the given date
    :return: a list date strings in form of YYYY-MM-DD
    """
    year_list, month_list, day_list = date_to_ymd(date=date, num_days=num_days)
    stop =0
    date_str_list = [str(y)+'-'+str(m).zfill(2)+'-'+str(d).zfill(2) for y, m, d in zip(year_list, month_list, day_list)]

    return date_str_list