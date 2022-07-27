import os
from pygnssr.common.utils.Equi7Grid import Equi7Grid, Equi7Tile, get_ftile_names
from pygnssr.common.utils.dcube import dcube_downsample
import numpy as np
from pygnssr.cygnss.CygnssDataCube import CygnssDataCube, read_cygnss_l1_dcube
from pygnssr.common.utils.netcdf_utils import compress_netcdf
from datetime import datetime
import click
import logging
import pickle
from functools import partial
import multiprocessing as mp
import json
import glob
from pygnssr.analytics.gnssr_analytics import cal_rfl, cal_nrfl, cal_slope, cal_drywet, cal_percentiles, cal_rssm, cal_cssm
from pygnssr.cygnss.process.L2.process_cygnss_l2_sm import copy_l1_vars_to_l2


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


def gen_l2_pars(ftile, start_date,  end_date, dir_work, dir_l1, dir_l2, dir_l2p, dir_smap_perc,
                agg_pix=1, gen_sm_flag=False, overwrite=False):
    try:
        p_stime = datetime.now()
        print('processing of ' + ftile + ' ....' )

        # create L2P datacube in working directory
        dc_l2p = CygnssDataCube(ftile, 'L2P', dir_work, flag='w')

        # get L1 datacube
        dc_l1 = read_cygnss_l1_dcube(ftile, dir_l1)

        #TODO apply time/date filter
        # if (sdate is not None) and (edate is not None):
        #       ind = np.where((dates >= stime) & (dates <= etime))
        #   ---------

        tile_l1 = Equi7Tile(ftile)
        # create indices of the L1 grid pixels
        m = 1  # m is set as 1 to have the same output grid size as input
        npix = int(tile_l1.shape[1] / m)
        y_indices = np.tile(range(0, npix), npix).reshape(npix, npix)
        x_indices = y_indices.transpose()

        # RFL calculation
        rfl = cal_rfl(dc_l1.nc, bias_corr=False)
        # read incidence angle
        inc = dc_l1.nc.variables['sp_inc_angle'][:, :, :]

        # Slope calculation
        slope, intcp = cal_slope(inc, rfl, x_indices, y_indices)

        # calculate Dry and Wet references
        # calculate normalized RFL at original resolution needed for dry and wet calculations
        r_slope = np.repeat(slope[np.newaxis], rfl.shape[0], axis=0)
        nrfl = cal_nrfl(inc, rfl, r_slope, ref_inc=40)

        # calculate nrfl statistics
        nrfl_mean = nrfl.mean(axis=0)
        nrfl_std = nrfl.std(axis=0)

        dry, wet = cal_drywet(nrfl, x_indices, y_indices)

        # calculate relative soil moisture RSSM  at original resolution (3km)
        r_dry = np.repeat(dry[np.newaxis], nrfl.shape[0], axis=0)
        r_wet = np.repeat(wet[np.newaxis], nrfl.shape[0], axis=0)
        rssm, pflag = cal_rssm(nrfl, r_dry, r_wet)

        # read smap percentiles
        file_smap_perc = os.path.join(dir_smap_perc, ftile[0:7], "SMAP_PERCENTILES_" + ftile + ".pkl")
        with open(file_smap_perc, 'rb') as f:
            perc_smap = pickle.load(f)

        perc_rssm = np.full((10, tile_l1.shape[0], tile_l1.shape[1]), np.nan, dtype=np.float32)
        for i in range(tile_l1.shape[0]):
            for j in range(tile_l1.shape[1]):
                perc_rssm[:, i, j] = cal_percentiles(rssm, i, j, nbins=10)

        # todo check if this part can be also moved to cal_drywet
        # convert param arrays to masked arrays, mask NaN values
        slope = np.ma.array(slope, mask=np.isnan(slope))
        intcp = np.ma.array(intcp, mask=np.isnan(intcp))
        dry = np.ma.array(dry, mask=np.isnan(dry))
        wet = np.ma.array(wet, mask=np.isnan(wet))
        perc_rssm = np.ma.array(perc_rssm, mask=np.isnan(perc_rssm))
        perc_smap = np.ma.array(perc_smap, mask=np.isnan(perc_smap))

        # write netCDF variables
        dc_l2p.nc['inc_slope'][:, :] = slope
        dc_l2p.nc['inc_intcp'][:, :] = intcp
        dc_l2p.nc['dry'][:, :] = dry
        dc_l2p.nc['wet'][:, :] = wet
        dc_l2p.nc['nrfl_mean'][:, :] = nrfl_mean
        dc_l2p.nc['nrfl_std'][:, :] = nrfl_std
        dc_l2p.nc['perc_rssm'][:, :, :] = perc_rssm[:, :, :]
        dc_l2p.nc['perc_smap'][:, :, :] = perc_smap[:, :, :]

        if gen_sm_flag:  #todo: replace this part with process_cygnss_l2_sm
            #todo methods from process_cygnss_l2_sm to do the job
            # downsample reflectivity and incidence angle
            ds_rfl = dcube_downsample(rfl, agg_pix)
            ds_inc = dcube_downsample(inc, agg_pix)
            # calculate the average of parameters during the downsampling
            ds_slope = dcube_downsample(slope, agg_pix, operation='mean')
            ds_dry = dcube_downsample(dry, agg_pix, operation='mean')
            ds_wet = dcube_downsample(wet, agg_pix, operation='mean')

            # determine output grid spacing
            res_l2 = agg_pix * tile_l1.res
            # define output tile name
            ftile_l2 = ftile[0:2] + str(res_l2).strip() + ftile[6:]

            # create L2 datacube
            dc_l2 = CygnssDataCube(ftile_l2, 'L2', dir_work, flag='w', l1_sample_file=dc_l1.fpath)

            # copy selected variables from L1 to L2 data file
            copy_l1_vars_to_l2(dc_l1.nc, dc_l2.nc, agg_pix=agg_pix)

            # write downsampled RFL
            dc_l2.nc['rfl'][0:ds_rfl.shape[0], :, :] = ds_rfl

            # calculate normalized RFL using downsampled parameters
            r_ds_slope = np.repeat(ds_slope[np.newaxis], ds_rfl.shape[0], axis=0)
            ds_nrfl = cal_nrfl(ds_inc, ds_rfl, r_ds_slope, ref_inc=40)
            dc_l2.nc['nrfl'][0:ds_nrfl.shape[0], :, :] = ds_nrfl

            # calculate relative soil moisture RSSM
            r_ds_dry = np.repeat(ds_dry[np.newaxis], ds_nrfl.shape[0], axis=0)
            r_ds_wet = np.repeat(ds_wet[np.newaxis], ds_nrfl.shape[0], axis=0)
            ds_rssm, pflag = cal_rssm(ds_nrfl, r_ds_dry, r_ds_wet)
            dc_l2.nc['rssm'][0:ds_rssm.shape[0], :, :] = ds_rssm
            dc_l2.nc['pflag'][0:ds_rssm.shape[0], :, :] = pflag

            cssm = cal_cssm(rssm, perc_rssm, perc_smap, x_indices, y_indices)
            ds_cssm = dcube_downsample(cssm, agg_pix)
            dc_l2.nc['cssm'][0:ds_cssm.shape[0], :, :] = ds_cssm

            dc_l2.close_nc()

            # todo move this up
            dst_l2_file = os.path.join(dir_l2, dc_l2.ftile.split('_')[0], os.path.basename(dc_l2.fpath))
            if os.path.exists(dst_l2_file) and not overwrite:
                os.remove(dc_l2.fpath)
                raise ValueError("Output file exists! Set overwrite or update keyword as True..." + dst_l2_file)
            else:
                os.makedirs(os.path.dirname(dst_l2_file), exist_ok=True)
                # compress l2 netcdf file and move to destination directory
                compress_netcdf(dc_l2.fpath, dst_l2_file)

        # close netcdf files
        dc_l1.close_nc()
        dc_l2p.close_nc()
        dst_l2p_file = os.path.join(dir_l2p, dc_l2p.ftile.split('_')[0], os.path.basename(dc_l2p.fpath))
        if os.path.exists(dst_l2p_file) and not overwrite:
            os.remove(dc_l2p.fpath)
            raise ValueError("Output file exists! Set overwrite or update keyword as True..." + dst_l2p_file)
        else:
            os.makedirs(os.path.dirname(dst_l2p_file), exist_ok=True)
            # compress the netcdf file and move to destination directory
            compress_netcdf(dc_l2p.fpath, dst_l2p_file)
            logging.info(ftile+"  processing successful! " + ": " + str(datetime.now()-p_stime))
            return True

    except Exception as e:
        logging.error(ftile + "  processing failed! ..", str(e))


def cygnss_param_wrapper(grid_res, dir_work, dir_dpool, dir_out=None, ftile_list=ftile_list,
                         gen_sm_flag=gen_sm_flag, sm_res=sm_res, sdate=sdate, edate=edate,
                         overwrite=overwrite, mp_num=mp_num):

    log_start_time = datetime.now()
    print(log_start_time, "L2 parameters calculation started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, log_start_time.strftime("%Y-%m-%d_%H%M%S") + "_CYGNSS_L2 ssm_calculation_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    if ftile_list is None:
        ftiles = get_ftile_names(dir_dpool, grid_res=grid_res, land=True, eq=True)
    else:
        ftiles = ftile_list

    dir_l1 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L1")
    dir_smap_perc = os.path.join(dir_dpool, "internal", "datacube", "smap_spl3smp_e", "percentiles")
    if dir_out is None:
        dir_l2 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2")
        dir_l2p = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2P")
    else:
        dir_l2 = os.path.join(dir_out, "L2")
        dir_l2p = os.path.join(dir_out, "L2P")

    edate = datetime.strptime(edate, "%Y-%m-%d")
    sdate = datetime.strptime(sdate, "%Y-%m-%d")
    agg_pix = int(int(sm_res)/int(grid_res))

    if mp_num == 1:
        for ftile in ftiles:
            gen_l2_pars(ftile, sdate,  edate, dir_work, dir_l1, dir_l2, dir_l2p, dir_smap_perc,
                        agg_pix=agg_pix, gen_sm_flag=gen_sm_flag,  overwrite=overwrite)
    else:
        prod_ftile = partial(gen_l2_pars, sdate=sdate,  edate=edate, dir_work=dir_work,
                             dir_l1=dir_l1, dir_l2=dir_l2, dir_l2p=dir_l2p, dir_smap_perc=dir_smap_perc,
                             agg_pix=agg_pix, gen_sm_flag=gen_sm_flag, overwrite=overwrite)
        p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)


@click.command()
@click.argument('grid_res', default='3000', type=str)
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--gen_sm', is_flag=True,
              help='If set the soil moisture data will be generated after parameters calculation')
@click.option('--sm_res', default='6000', type=int,
              help='It will be used for output sm grid resolution if gen_sm is set')
@click.option('--sdate', default='2017-04-01', type=str,
              help='Start date in following format: "%Y-%m-%d" ')
@click.option('--edate', default='2020-04-01', type=str,
              help='End date in following format: "%Y-%m-%d" ')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dir_work, dir_dpool, dir_out, gen_sm, sm_res, sdate, edate, overwrite, mp_num):
    """
    This program calculate CYGNSS geophysical parameters

    :param grid_res:
    """
    ftile_list = None
    cygnss_param_wrapper(grid_res, dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list,
                         gen_sm_flag=gen_sm, sm_res=sm_res, sdate=sdate, edate=edate,
                         overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()
