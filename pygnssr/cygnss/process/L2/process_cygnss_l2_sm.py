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
from pygnssr.analytics.gnssr_analytics import cal_rfl, cal_nrfl, cal_rssm, cal_cssm


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


def gen_l2_sm(ftile, dir_work, dir_l1, agg_pix=1,
              dir_l2=None, dir_l2p=None, overwrite=False):

    try:
        p_stime = datetime.now()
        print('processing of ' + ftile + ' ....' )

        tile_l1 = Equi7Tile(ftile)
        # determine output grid spacing
        res_l2 = int(agg_pix * tile_l1.res)
        # define output tile name
        ftile_l2 = ftile[0:2] + str(res_l2).strip() + ftile[6:]

        dst_dc_l2 = CygnssDataCube(ftile_l2, 'L2', os.path.join(dir_l2, ftile_l2.split('_')[0]))
        if os.path.exists(dst_dc_l2.fpath) and not overwrite:
            raise Exception("Output file exists! Set overwrite keyword as True..." + dst_dc_l2.fpath)

        # get L1 datacube
        dc_l1 = read_cygnss_l1_dcube(ftile, dir_l1)

        # get L2P parameters datacube
        dc_l2p = read_cygnss_l2p_dcube(ftile, dir_l2p)
        # read required parameters
        slope = dc_l2p.nc['inc_slope'][:, :]
        dry = dc_l2p.nc['dry'][:, :]
        wet = dc_l2p.nc['wet'][:, :]
        perc_rssm = dc_l2p.nc['perc_rssm'][:, :, :]
        perc_smap = dc_l2p.nc['perc_smap'][:, :, :]

        # RFL calculation
        rfl = cal_rfl(dc_l1.nc, bias_corr=False)
        # read incidence angle
        inc = dc_l1.nc.variables['sp_inc_angle'][:, :, :]

        # calculate normalized RFL at original resolution (3km)
        r_slope = np.repeat(slope[np.newaxis], rfl.shape[0], axis=0)
        nrfl = cal_nrfl(inc, rfl, r_slope, ref_inc=40)

        # calculate relative soil moisture RSSM  at original resolution (3km)
        r_dry = np.repeat(dry[np.newaxis], nrfl.shape[0], axis=0)
        r_wet = np.repeat(wet[np.newaxis], nrfl.shape[0], axis=0)
        rssm, pflag = cal_rssm(nrfl, r_dry, r_wet)

        # downsample reflectivity and incidence angle
        ds_rfl = dcube_downsample(rfl, agg_pix)
        ds_inc = dcube_downsample(inc, agg_pix)
        # calculate the average of parameters during the downsampling
        ds_slope = dcube_downsample(slope, agg_pix, operation='mean')
        ds_dry = dcube_downsample(dry, agg_pix, operation='mean')
        ds_wet = dcube_downsample(wet, agg_pix, operation='mean')

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

        # calculate relative soil moisture (RSSM)
        r_ds_dry = np.repeat(ds_dry[np.newaxis], ds_nrfl.shape[0], axis=0)
        r_ds_wet = np.repeat(ds_wet[np.newaxis], ds_nrfl.shape[0], axis=0)
        ds_rssm, pflag = cal_rssm(ds_nrfl, r_ds_dry, r_ds_wet)
        dc_l2.nc['rssm'][0:ds_rssm.shape[0], :, :] = ds_rssm
        dc_l2.nc['pflag'][0:ds_rssm.shape[0], :, :] = pflag

        # create indices of the L1 grid pixels
        m = 1  # m is set as 1 to have the same output grid size as input
        npix = int(tile_l1.shape[1] / m)
        y_indices = np.tile(range(0, npix), npix).reshape(npix, npix)
        x_indices = y_indices.transpose()

        cssm = cal_cssm(rssm, perc_rssm, perc_smap, x_indices, y_indices)
        ds_cssm = dcube_downsample(cssm, agg_pix)
        dc_l2.nc['cssm'][0:ds_cssm.shape[0], :, :] = ds_cssm

        # close netcdf files
        dc_l1.close_nc()
        dc_l2.close_nc()
        dc_l2p.close_nc()

        # todo move this up
        # get destination file, check if it is already available
        dst_l2_file = os.path.join(dir_l2, dc_l2.ftile.split('_')[0], os.path.basename(dc_l2.fpath))
        if os.path.exists(dst_l2_file) and not overwrite:
            os.remove(dc_l2.fpath)
            raise ValueError("Output file exists! Set overwrite or update keyword as True..." + dst_l2_file)
        else:
            os.makedirs(os.path.dirname(dst_l2_file), exist_ok=True)
            # compress l2 netcdf file and move to destination directory
            compress_netcdf(dc_l2.fpath, dst_l2_file)
            logging.info(ftile+"  processing successful! " + ": " + str(datetime.now()-p_stime))
            return True

    except Exception as e:
        print("Failed to process:  " + ftile, e)
        logging.error(ftile + "  processing failed! .." + "\n" + str(e))
        return False


def copy_l1_vars_to_l2(nc_l1, nc_l2, agg_pix=1):
    var_names = ['spacecraft_num', 'ddm_timestamp_utc', 'sp_lon', 'sp_lat',
                 'prn_code', 'sv_num', 'sp_inc_angle', 'quality_flags']
    for vname in var_names:
        ds = dcube_downsample(nc_l1.variables[vname][:, :, :], agg_pix)
        tsize = ds.shape[0]
        nc_l2[vname][0:tsize, :, :] = ds


def cygnss_sm_wrapper(grid_res, dir_work, dir_dpool, dir_out=None, ftile_list=None,
                      out_grid_res=None, overwrite=False, mp_num=1):

    log_start_time = datetime.now()
    print(log_start_time, " L2 soil moisture calculation started from python code ...")
    # setup logging ----------------------------------------------------------------------------------------
    log_file = os.path.join(dir_work, log_start_time.strftime("%Y-%m-%d_%H%M%S") + "_CYGNSS_L2 ssm_calculation_log_file.log")
    log_level = logging.INFO
    log_frmt = '%(asctime)s [%(levelname)s] - %(message)s'
    log_datefrmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(filename=log_file, filemode='w', level=log_level, format=log_frmt, datefmt=log_datefrmt)
    # setup logging ----------------------------------------------------------------------------------------

    if out_grid_res is None:
        out_grid_res = int(grid_res)
    elif int(out_grid_res) not in [3000, 6000, 12000, 24000, 30000, 40000]:
        raise ValueError("Given output grid resolution is not supported!")

    agg_pix = int(int(out_grid_res)/int(grid_res))

    if ftile_list is None:
        ftiles = get_ftile_names(dir_dpool, grid_res=grid_res, land=True, eq=True)
    else:
        ftiles = ftile_list

    dir_l1 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L1")
    dir_l2p = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2P")
    if dir_out is None:
        dir_l2 = os.path.join(dir_dpool, "internal", "datacube", "cygnss", "dataset", "L2")
    else:
        dir_l2 = os.path.join(dir_out)

    if mp_num == 1:
        for ftile in ftiles:
            gen_l2_sm(ftile, dir_work, dir_l1,
                      agg_pix=agg_pix, dir_l2=dir_l2, dir_l2p=dir_l2p, overwrite=overwrite)
    else:
        prod_ftile = partial(gen_l2_sm, dir_work=dir_work, dir_l1=dir_l1,
                             agg_pix=agg_pix, dir_l2=dir_l2, dir_l2p=dir_l2p, overwrite=overwrite)
        p = mp.Pool(processes=mp_num).map(prod_ftile, ftiles)

    logging.info("============================================")
    logging.info("Total processing time "+str(datetime.now()-log_start_time))
    logging.shutdown()
    print(datetime.now(), " CYGNSS L2 soil moisture calculation is finished!")


@click.command()
@click.argument('grid_res', default='3000', type=str)
@click.option('--dir_work', default=r"/home/ubuntu/_working_dir", type=str,
              help='Working directory to store intermediate results.')
@click.option('--dir_dpool', default=r"/home/ubuntu/datapool", type=str,
              help='DataPool directory ("datapool in gnssr S3 bucket")')
@click.option('--dir_out', default=None, type=str,
              help='Destination directory. Default is the internal datacube in "gnssr S3 bucket"')
@click.option('--out_grid_res', type=int,
              help='if set different than grid_res, it will be used for output grid resolution')
@click.option('--overwrite', is_flag=True,
              help='if set, the output data files will be overwritten')
@click.option('--mp_num', default=8, type=int,
              help='Number of workers to be used for multi-processing')
def main(grid_res, dir_work, dir_dpool, dir_out, out_grid_res, overwrite, mp_num):
    """
    This program calcualte CYGNSS surface soil moisture from resampled L1 data using L2P parameters

    :param grid_res: Input grid spatial resolution (L1 and L2ß grid spatial resolutions)
    """
    ftile_list = None
    cygnss_sm_wrapper(grid_res, dir_work, dir_dpool, dir_out=dir_out, ftile_list=ftile_list,
                      out_grid_res=out_grid_res, overwrite=overwrite, mp_num=mp_num)


if __name__ == "__main__":
    main()

