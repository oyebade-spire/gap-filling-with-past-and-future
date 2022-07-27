import datetime
import numpy as np

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"



def get_time_intervals(stime, etime, interval_type=None, climatologic=False):

    """

    Parameters
    ----------
        stime: Start date in datetime format (required)

        etime: End date in datetime format (required)      
        
        interval_type: String from following list:
                        6h: 6 hourly intervals starting at [3, 9, 15, 21] and ending at [9, 15, 21, 3(next day)]
                            So the centreal hour will be [6, 12, 18, 0]
                        daily: daily starting and ending at 0 hour of the day
                        decadal: 10-days periods
                        monthly: starting at the first day of month, ending at the end of the month
                        seasonally: the time periods will be as following:
                                    (Mar-Apr-May, Jun-Jul-Aug, Sep-Oct-Nov, Dec-Jan-Feb)
                        (default is None, returning a single interval starting with stime and ending with etime)
            
        climatologic: Boolean (optional)
            If set together with monthly or seasonal keyword, a list of arrays 
            of stime and etime will be returned after grouping over multiple years

    Returns
    ------- 
        It returns time intervals (starting and end dates of the intervals) between given start and end time

    """

    if interval_type is not None and interval_type.lower() != '6h':
        # extend start time to first hour in that day
        stime = stime.replace(hour=0, minute=0, second=0, microsecond=0)
        # extend end time to last hour in that day
        if (etime.hour != 0 and etime.minute != 0 and etime.second != 0):
            etime = etime.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
    else:
        stime = stime.replace(minute=0, second=0, microsecond=0)
        etime = etime.replace(minute=0, second=0, microsecond=0)



    # initiate start and end date-arrays with given start date
    stime_arr = [stime]
    etime_arr = [etime]

    if interval_type is not None:
        if interval_type.lower() == '6h':
            # find the closest hour to [3, 9, 15, 21]
            h_arr = np.array([3, 9, 15, 21])
            idx = abs(stime.hour - h_arr).argmin()
            stime_arr = [stime.replace(hour=h_arr[idx])]
            etime_arr = [stime_arr[0] + datetime.timedelta(hours=6)]
            while etime_arr[-1] < etime:
                stime_arr.append(stime_arr[-1] + datetime.timedelta(hours=6))
                etime_arr.append(etime_arr[-1] + datetime.timedelta(hours=6))
        elif interval_type.lower() == 'daily':
            stime_arr = [stime]
            etime_arr = [stime_arr[0] + datetime.timedelta(days=1)]
            while etime_arr[-1] < etime:
                stime_arr.append(stime_arr[-1] + datetime.timedelta(days=1))
                etime_arr.append(etime_arr[-1] + datetime.timedelta(days=1))
        elif interval_type.lower() == 'decadal':
            sdate, edate = _first_last_day_of_decade(stime)
            stime_arr = [sdate]
            etime_arr = [edate]
            while etime_arr[-1] < etime:
                stime_arr.append(etime_arr[-1] + datetime.timedelta(days=1))
                sdate, edate = _first_last_day_of_decade(stime_arr[-1])
                etime_arr.append(edate)
        elif interval_type.lower() == 'monthly':
            etime_arr = [_last_day_of_month(stime)]
            # extend the first start time to the first day of month
            stime_arr = [etime_arr[0].replace(day=1)]
            while etime_arr[-1] < etime:
                stime_arr.append(etime_arr[-1] + datetime.timedelta(days=1))
                etime_arr.append(_last_day_of_month(stime_arr[-1]))
        elif interval_type.lower() == 'seasonally':
            # extend the first start time to the first day of month
            last_day_stime =_last_day_of_month(stime)
            stime_arr = [datetime.datetime(last_day_stime.year, last_day_stime.month, 1)]
            # season start month
            s_month_arr = [3, 6, 9, 12]
            # give a code between 0-3 to each of 12 months indicating the season
            season_codes = [3, 3, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
            s_month = s_month_arr[season_codes[stime.month - 1]]
            # find the end of the first period
            if stime.month in [1, 2]:
                stime_arr[0] = stime.replace(year=stime.year - 1, month=s_month, day=1)
            else:
                stime_arr[0] = stime.replace(month=s_month, day=1)
            etime_arr[0] = _last_day_of_month(stime_arr[0] + datetime.timedelta(days=80))
            while etime_arr[-1] < etime:
                stime_arr.append(etime_arr[-1] + datetime.timedelta(days=1))
                temp = stime_arr[-1] + datetime.timedelta(days=80)  # someday in 3 month later
                etime_arr.append(_last_day_of_month(temp))
        else:
            raise ValueError("Interval type is not supported!")

        if climatologic:
            # create temporary arrays
            s_arr = np.array(stime_arr)
            e_arr = np.array(etime_arr)
            # group stime_arr based on month number
            months = np.empty(0, dtype='int16')
            stime_arr = []
            etime_arr = []
            for date in s_arr:
                months = np.append(months, date.month)
            if interval_type.lower() == 'monthly':
                uniq_months = list(set(months))
                for um in uniq_months:
                    stime_arr.append(s_arr[months == um].tolist())
                    etime_arr.append(e_arr[months == um].tolist())
            elif interval_type.lower() == 'seasonally':
                for um in s_month_arr:
                    stime_arr.append(s_arr[months == um].tolist())
                    etime_arr.append(e_arr[months == um].tolist())
            else:
                raise Exception('For calculating of climatological values, interval_type keyword should be set'
                                'as either monthly or seasonal')

    return stime_arr, etime_arr


def _last_day_of_month(any_date):
    next_month = any_date.replace(day=28) + datetime.timedelta(days=4)  # this will never fail
    return next_month - datetime.timedelta(days=next_month.day)

def _first_last_day_of_decade(any_date):
    if any_date.day < 10 :
        sdate = any_date.replace(day=1)
        edate = any_date.replace(day=10)
    elif (any_date.day >= 10)and(any_date.day < 20):
        sdate = any_date.replace(day=11)
        edate = any_date.replace(day=20)
    else:
        sdate = any_date.replace(day=21)
        next_month = any_date.replace(day=28) + datetime.timedelta(days=4)
        edate = next_month - datetime.timedelta(days=next_month.day)
    return sdate, edate

def main():
    stime = datetime.datetime(2020, 7, 21)
    etime = datetime.datetime(2020, 9, 1)

    date_int = get_time_intervals(stime, etime, interval_type=None)
    print(date_int)

if __name__ == "__main__":
    main()
