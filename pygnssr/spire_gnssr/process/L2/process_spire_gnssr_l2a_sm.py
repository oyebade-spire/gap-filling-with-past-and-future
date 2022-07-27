import os
import numpy as np
import click
import logging
import glob
import json
import multiprocessing as mp
from functools import partial
from datetime import datetime, timedelta
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
from pygnssr.common.utils.dcube import dcube_downsample
from pygnssr.cygnss.CygnssDataCube import CygnssDataCube, read_cygnss_l1_dcube, read_cygnss_l2p_dcube
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from pygnssr.analytics.gnssr_analytics import cal_rfl, cal_nrfl, cal_rssm, gen_cdf_match
from netCDF4 import Dataset, num2date, date2num
from pygnssr.common.utils.dcube import get_e7idx


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"

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

    # list of variables for each reflection
    v1 = {'sample_time': np.float64,
          'sp_lat': np.float32,
          'sp_lon': np.float32,
          'quality_flags': np.uint64,
          'tx_system': np.int16,
          'reflectivity_at_sp': np.float64,
          'track_id': np.int64,
          'sp_incidence_angle': np.float32,
          'tx_svn': np.int16}


    v2 = { 'smap_sm': {'dtype': np.float32,
                       'fill_value': '-9999.0',
                       'attrs': {'long_name': 'SMAP surface soil moisture',
                                 'units': 'cm³/cm³',
                                 'comment': 'tobe filled later'}},

           'smap_qflag': {'dtype': np.uint16,
                          'fill_value': '0',
                          'attrs': {'long_name': 'Data retrieval quality flag',
                                    'comment': 'tobe filled later'}},

           'smap_sflag': {'dtype': np.uint16,
                          'fill_value': '0',
                          'attrs': {'long_name': 'Surface condition flag',
                                    'comment': 'tobe filled later'}},

           'rssm': {'dtype': np.float32,
                    'fill_value': '-9999.0',
                    'attrs': {'long_name': 'relative surface soil moisture',
                              'units': '%',
                              'comment': 'uncalibrated relative surface soil moisture ranging between '
                                         '0 and 100'}},

           'cssm': {'dtype': np.float32,
                    'fill_value': '-9999.0',
                    'attrs': {'long_name': 'calibrated surface soil moisture',
                              'units': 'cm³/cm³',
                              'comment': 'surface soil moisture after calibration with auxiliary data in '
                                         'volumetric units'}},
           #todo: set the flag bitwise in next release
           'pflag': {'dtype': np.int32,
                     'fill_value': '-9999',
                     'attrs': {'long_name': 'L2 processing quality flag',
                               'flag_masks': '[0, 1, 2]',
                               'flag_meaning': '0: no correction, '
                                               '1: reserved, TBD'
                                               '2: reserved, TBD',
                               'comment': 'Processing flags provide information about'
                                          'masking or corrections applied on data products'}}}




    return v1, v2


def _create_nc_out(file_dst, file_src):

    nc_src = Dataset(file_src, 'r')
    # output nc
    nc = Dataset(file_dst, 'w', diskless=True, persist=True)

    # todo: generalize description, mention the name of data and paramters used for sm retrieval
    nc.history = 'Created on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')
    nc.creator = 'SPIRE GLOBAL'
    nc.description = 'Spire Level-2a soil moisture product'
    nc.source = 'Calculated from GNSS-R level-1a data'
    nc.version = '0.1'
    nc.createDimension('sample', nc_src.dimensions['sample_time'].size)

    v1, v2 = _get_template()

    for p, q in v1.items():
        v1[p] = nc.createVariable(p, q, ('sample'))
        nc[p].setncatts(nc_src[p].__dict__)

    for p in v2.keys():
        # copy attributes
        at = v2[p].copy()
        v2[p] = nc.createVariable(p, at['dtype'], ('sample'), fill_value=at['fill_value'])
        nc[p].setncatts(at['attrs'])

    nc_src.close()
    return nc


def _get_smap_time_mask(curr_time, nctime_arr, latency=12):

    st = date2num(curr_time - timedelta(hours=latency), units=nctime_arr.units, calendar=nctime_arr.calendar)
    en = date2num(curr_time + timedelta(hours=latency), units=nctime_arr.units, calendar=nctime_arr.calendar)
    val_t_smap = np.ma.masked_greater_equal(np.ma.masked_less(nctime_arr, st), en)

    return val_t_smap.mask


def gen_l2a_sm(file_idx, dir_work, dir_l1, dir_l2p, dir_comb_ssm_l2u1, dir_out, tracklist = None, overwrite=False, rfl_offset=0):

    try:
        p_stime = datetime.now()

        fname_l1 = os.path.basename(file_idx).split('__')[0]
        # TODO: generalize this part, adapt according to cygnss file name
        fname_l2 = fname_l1.replace("Spire_GNSSR_L1a_", "Spire_GNSSR_L2a_")
        date_name = fname_l1[26:36]
        sat_name = fname_l1[16:21]
        file_src = os.path.join(dir_l1, sat_name, date_name, fname_l1)
        file_temp = os.path.join(dir_work, fname_l2)
        file_dst= os.path.join(dir_out, sat_name, date_name, fname_l2)

        print('processing of ' + fname_l1 + ' ....' )
        if os.path.exists(file_dst) and not overwrite:
            print('File exists! set overwrite keyword!')
            return False

        # initialize output netcdf file
        nc = _create_nc_out(file_temp, file_src)

        # read input data
        nc_src = Dataset(file_src, 'r')
        rfl = 10.0 * np.log10(nc_src.variables['reflectivity_at_sp'][:]) + rfl_offset
        inc = nc_src.variables['sp_incidence_angle'][:]
        qflags = nc_src.variables['quality_flags'][:]
        track_ids = nc_src.variables['track_id'][:]
        sp_lon = nc_src.variables['sp_lon'][:].astype(np.float64)
        sp_lat = nc_src.variables['sp_lat'][:].astype(np.float64)
        acq_stime = num2date(nc_src.variables['sample_time'][0], nc_src.variables['sample_time'].units, nc_src.variables['sample_time'].calendar)
        # time period filter for smap sm
        smap_latency = 24
        #todo copy v1 variables from l1a to l2a data product
        v1, v2 = _get_template()
        for vname in v1.keys():
            nc.variables[vname][:] = nc_src.variables[vname][:]

        nc_src.close()

        ft_arr, ix_arr, iy_arr= _get_indices(file_idx)

        ft_old = "null"
        for i, (ft, x, y) in enumerate(zip(ft_arr, ix_arr, iy_arr)):
            # TEMP -----------------
            if (tracklist is not None) and (str(track_ids[i]) not in tracklist):
                    continue
            # TEMP -----------------
            if ft != ft_old:
                try:
                    # get L2P parameters
                    # todo: set conditions for reading corresponding paramter datacube
                    sgrid = ft.split('_')[0]
                    dc_l2p = CygnssDataCube(ft, 'L2P', os.path.join(dir_l2p, sgrid), flag='r')
                    # read required parameters
                    slope = dc_l2p.nc['inc_slope'][:, :]
                    dry = dc_l2p.nc['dry'][:, :]
                    wet = dc_l2p.nc['wet'][:, :]
                    perc_rssm = dc_l2p.nc['perc_rssm'][:, :, :]
                    perc_smap = dc_l2p.nc['perc_smap'][:, :, :]
                    dc_l2p.close_nc()

                    # get smap and cygnss soil moisture
                    nc_comb = get_dcube(ft.replace('3000', '6000'), 'comb-ssm', dir_comb_ssm_l2u1, data_level='l2u1')

                    if nc_comb is not None:
                        #cygns_rssm = nc_comb['cygnss']['rssm'][:, :, :]
                        #cygns_cssm = nc_comb['cygnss']['cssm'][:, :, :]

                        smap_sm = nc_comb['smap']['sm'][:, :, :]
                        #smap_qflag = nc_comb['smap']['retrieval_qual_flag'][:, :, :]
                        #smap_sflag = nc_comb['smap']['surface_flag'][:, :, :]

                        # get valid indices within the given time period
                        smap_time_mask = _get_smap_time_mask(acq_stime, nc_comb['smap']['time_utc'], latency=smap_latency)
                        smap_sm = np.ma.masked_where(smap_time_mask, smap_sm)

                except Exception as e:
                    print(str(e))
                ft_old = ft

            # calculate soil moisture
            try:
                # normalize reflectivity using the slope parameter
                if not rfl.mask[i]:
                    nrfl = np.array(cal_nrfl(inc[i], rfl[i], slope[x, y], ref_inc=40))
                    # calcualte the relative reflectivtiy (relative soil moisture)
                    rssm, nc.variables['pflag'][i]  = cal_rssm(nrfl, dry[x, y], wet[x, y])
                    nc.variables['rssm'][i] = rssm
                    psrc = perc_rssm[:, x, y].flatten()
                    pref = perc_smap[:, x, y].flatten()

                    if not rssm.mask and not (np.any(psrc.mask) or np.any(pref.mask)):
                        cssm = gen_cdf_match(rssm, psrc, pref, k=1)
                        nc.variables['cssm'][i] = cssm

                    # smap data resampling
                    # following line is still valid and usable, since the 3000 and 6000 tiles are both T6 type tile
                    #_, xx, yy = get_e7idx(6000, sp_lon[i], sp_lat[i])
                    nc.variables['smap_sm'][i] = smap_sm[:, int(x/2), int(y/2)].mean()


            except Exception as e:
                print(str(e))

        nc.close()

        # compress the netcdf file and move to destination directory
        os.makedirs(os.path.dirname(file_dst), exist_ok=True)
        compress_netcdf(file_temp, file_dst)

    except Exception as e:
        print(str(e))


def sm_l2a_wrapper(grid_res, dname_l1, dname_l2p, dir_work, dir_dpool, dir_out=None,
                   stime=None, etime=None, days=14, overwrite=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, dname_l2p.upper() + " L2A soil moisture calculation started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, log_start_time.strftime("%Y-%m-%d_%H%M%S") +'_'+ dname_l1.upper() + "_L2A_ssm_calculation_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    if etime is None:
        etime = datetime.now()
    else:
        etime = datetime.strptime(etime, "%Y-%m-%d %H:%M:%S")
    if stime is None:
        stime = etime - timedelta(days)
    else:
        stime = datetime.strptime(stime, "%Y-%m-%d %H:%M:%S")

    if dname_l1.lower() == 'cygnss':
        dir_l1 = os.path.join(dir_dpool, "external", "cygnss", "L1", "v2.1")
        dir_e7grid_idx = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "cygnss_e7_indices")
    elif dname_l1.lower() == 'spire_gnssr':
        rfl_offset = 10  # set this to remove the bias between spire gnss-r and cygnss
        dir_l1 = os.path.join(dir_dpool, "internal", "spire_gnssr", "prod-0.3.7", "l1")
        dir_e7grid_idx = os.path.join(dir_dpool, "internal", "datacube", "spire_gnssr", "prod-0.3.7", "spire_gnssr_e7_indices")
    else:
        raise ValueError('The given dname_l1 is not supported!')

    dir_comb_ssm_l2u1 = os.path.join(dir_dpool, "internal", "datacube", "comb_ssm", "dataset", "L2U1")
    dir_l2p = os.path.join(dir_dpool, "internal", "datacube", dname_l2p.lower(), "dataset", "L2P")
    if dir_out is None:
        dir_out = os.path.join(dir_dpool, "internal", "temp_working_dir", "2020-02-22_spire_gnsr_l2a_sm", dname_l1.lower())


    all_idx_files = glob.glob(os.path.join(dir_e7grid_idx, "*/*__e7indices_"+str(grid_res)+"m.json"))
    # TODO: tobe adapted for cygnss index files
    dates = np.array([datetime.strptime(os.path.basename(x)[26:36], "%Y-%m-%d") for x in all_idx_files])
    # filter dates
    if (stime is not None) and (etime is not None):
        ind = np.where((dates >= stime) & (dates <= etime))
        files_idx_in = list(np.array(all_idx_files)[ind])

    #todo: implement multi processing
    for file_idx_in in files_idx_in:
        ind = np.where(np.array(files_idx) == file_idx_in)
        tracklist = list(np.array(all_tracks)[ind])
        gen_l2a_sm(file_idx_in, dir_work, dir_l1, dir_l2p, dir_comb_ssm_l2u1, dir_out, tracklist=tracklist, overwrite=overwrite, rfl_offset=rfl_offset)

    logging.info("============================================")
    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    print(datetime.now(), dname_l2p.upper() + " GNSS-R L2A soil moisture calculation is finished!")



@click.command()
@click.argument('grid_res', default='3000', type=str)
@click.option('--dname_l1', default="spire_gnssr", type=str,
              help='The name of Level-1 input dataset (e.g. "cygnss", "spire_gnssr"')
@click.option('--dname_l2p', default="cygnss", type=str,
              help='The name of Level-2 input paramters (e.g. "cygnss", "spire_gnssr"')
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--days', default=14, type=int,
              help='Number of days in the past to be used as time filter for searching '
                   'and downloading the data files. Default is 14 days')
@click.option('--stime', default=None, type=str,
              help='Start date and time in following format: "%Y-%m-%d %H:%M:%S"  Default is None. '
                   'If provided, then the "days" argument will be overridden!')
@click.option('--etime', default=None, type=str,
              help='End date and time in following format: "%Y-%m-%d %H:%M:%S"  Default is None.'
                   'If not provided, then current date will be used as end time')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dname_l1, dname_l2p, dir_work, dir_dpool, dir_out, days, stime, etime, overwrite, mp_num):


    """
    This program generates trackwise soil moisture from spire_gnssr level-1 orbital tracks
    using given theparamter datacube

    :param grid_res: grid spacing used for reading the corresponsing parameters datacube, default is 3000 meter
    """
    stime = datetime(2020, 11, 28).strftime("%Y-%m-%d %H:%M:%S")
    etime = datetime(2020, 11, 29).strftime("%Y-%m-%d %H:%M:%S")
    mp_num = 1
    overwrite = True
    sm_l2a_wrapper(grid_res, dname_l1, dname_l2p, dir_work, dir_dpool, dir_out=dir_out,
                   stime=stime, etime=etime, days=days, overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()

