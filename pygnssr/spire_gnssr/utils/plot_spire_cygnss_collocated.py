import os
import glob
from netCDF4 import Dataset, num2date, date2num
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
from mypy.utils.misc import warp_lc
from pygnssr.common.utils.gdalport import read_tiff
import multiprocessing as mp
from functools import partial
from scipy.stats import gaussian_kde

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"

def _get_cygnss_mask(t_spire, t_cygnss, latency=12):
    mask = np.ma.getmaskarray(t_spire)
    st = date2num(num2date(t_spire[~mask].min(), units=t_spire.units, calendar=t_spire.calendar)-
                  timedelta(hours=latency), units=t_cygnss.units, calendar=t_cygnss.calendar)

    en = date2num(num2date(t_spire[~mask].max(), units=t_spire.units, calendar=t_spire.calendar)+
                  timedelta(hours=latency), units=t_cygnss.units, calendar=t_cygnss.calendar)

    val_t_cygnss = np.ma.masked_greater_equal(np.ma.masked_less(t_cygnss, st), en)
    return val_t_cygnss.mask



def _plot_basemap(lon_arr, lat_arr, map_name=None, extent=None):
    if  extent is None:
        view='global'
        extent = [-180, -90, 180, 90]
    else:
        view='regional'
    # create map using BASEMAP
    m = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[1], urcrnrlon=extent[2], urcrnrlat=extent[3], resolution='l',
                lat_0=(extent[3] - extent[1]) / 2, lon_0=(extent[2] - extent[0]) / 2, projection='cyl', area_thresh=10000.)
    if map_name is not None:
        getattr(m, map_name)()
    m.scatter(lon_arr, lat_arr, marker='o', c='red', s=1)
    if view == 'regional':
        m.scatter(lon_arr[0], lat_arr[0], marker='>', c='red', s=50)
    else:
        m.scatter(lon_arr[0], lat_arr[0], marker='o', c='red', s=10)

def _plot_lc(lon_arr, lat_arr, extent, dirs, temp_lc_name=None):
    file_lc = os.path.join(dirs['lc'] , "ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif")
    lc_temp = warp_lc(extent, file_lc, dirs['gdal_path'], dirs['work'], out_file_name=temp_lc_name)
    img = plt.imread(lc_temp)
    # create map using BASEMAP
    m = Basemap(llcrnrlon=extent[0], llcrnrlat=extent[1], urcrnrlon=extent[2], urcrnrlat=extent[3], resolution='l',
                lat_0=(extent[3] - extent[1]) / 2, lon_0=(extent[2] - extent[0]) / 2, projection='cyl', area_thresh=10000.)
    m.imshow(img, origin='upper')

    m.scatter(lon_arr, lat_arr, marker='o', c='red', s=0.5)
    m.scatter(lon_arr[0], lat_arr[0], marker='>', c='red', s=50)

    # clean working directory
    temp_file = os.path.join(dirs['work'], temp_lc_name+'.tif')
    if os.path.exists(temp_file):
        os.remove(temp_file)

def _get_lc_code(ftile, dir_lc_dcube, ix_arr, iy_arr):
    sgrid = ftile.split('_')[0]
    grid_res = sgrid[2:-1]
    # read landcover data
    lc_file = os.path.join(dir_lc_dcube, "EQUI7_"+ str(grid_res)+"M", sgrid, "ESACCI-LC_"+str(ftile)+'.tif')
    lc_tile, _ = read_tiff(lc_file)
    lc_tile = np.rot90(lc_tile, k=-1)
    return lc_tile[ix_arr, iy_arr]


def _read_track_variables(file_spire, file_cygnss):
    spire = {}
    cygnss = {}

    nc_spire = Dataset(file_spire, 'r')
    spire['track_ids'] = nc_spire.variables['track_id'][:]
    spire['t'] = nc_spire.variables['sample_time']
    spire['snr'] = nc_spire.variables['reflect_snr_at_sp'][:]
    spire['rfl'] = 10.0 * np.log10(nc_spire.variables['reflectivity_at_sp'][:])
    spire['gain'] = nc_spire.variables['rx_ant_gain_reflect'][:]
    spire['cgain'] = spire['gain'] - nc_spire.variables['rx_ant_gain_correction'][:]
    spire['inc'] = nc_spire.variables['sp_incidence_angle'][:]
    spire['sp_lon'] = nc_spire.variables['sp_lon'][:]
    spire['sp_lat'] = nc_spire.variables['sp_lat'][:]
    spire['qflags'] = nc_spire.variables['quality_flags'][:]
    spire['pflags'] = nc_spire.variables['processing_config'][:]

    #------------------------------------------------

    nc_cygnss = Dataset(file_cygnss, 'r')
    cygnss['t'] = nc_cygnss.variables['ddm_timestamp_utc']
    cygnss['snr'] = 10.0 * np.log10(10 ** (nc_cygnss.variables['ddm_snr'][:, :] / 10.0) - 1.0)
    cygnss['inc'] = nc_cygnss.variables['sp_inc_angle'][:, :]
    cygnss['gain'] = nc_cygnss.variables['sp_rx_gain'][:, :]
    cygnss['rfl'] = nc_cygnss.variables['rfl'][:, :]
    cygnss['brcs_sp'] = 10.0 * np.log10(nc_cygnss.variables['brcs_sp'][:, :])
    cygnss['brcs_peak'] = 10.0 * np.log10(nc_cygnss.variables['brcs_peak'][:, :])

    # get cygnss mask base on cygnss and spire nc_time
    cygnss_mask = _get_cygnss_mask(spire['t'], cygnss['t'], latency=24)
    cygnss['snr_rt'] = np.ma.masked_where(cygnss_mask, cygnss['snr']).mean(axis=1)
    cygnss['rfl_rt'] = np.ma.masked_where(cygnss_mask, cygnss['rfl']).mean(axis=1)
    cygnss['brcs_sp_rt'] = np.ma.masked_where(cygnss_mask, cygnss['brcs_sp']).mean(axis=1)
    cygnss['brcs_peak_rt'] = np.ma.masked_where(cygnss_mask, cygnss['brcs_peak']).mean(axis=1)
    cygnss['gain_rt'] = np.ma.masked_where(cygnss_mask, cygnss['gain']).mean(axis=1)
    cygnss['inc_rt'] = np.ma.masked_where(cygnss_mask, cygnss['inc']).mean(axis=1)

    cygnss['snr_mean'] = cygnss['snr'].mean(axis=1)
    cygnss['rfl_mean'] = cygnss['rfl'].mean(axis=1)
    cygnss['brcs_sp_mean'] = cygnss['brcs_sp'].mean(axis=1)
    cygnss['brcs_peak_mean'] = cygnss['brcs_peak'].mean(axis=1)
    cygnss['gain_mean'] = cygnss['gain'].mean(axis=1)
    cygnss['inc_mean'] = cygnss['inc'].mean(axis=1)

    return spire, cygnss


def _plot_file(file_cygnss, dirs):
    fname_spire = os.path.basename(file_cygnss)[18:]
    date_name = fname_spire[26:36]
    sat_name = fname_spire[16:21]
    file_spire = os.path.join(dirs['spire'], sat_name, date_name, fname_spire)

    spire, cygnss = _read_track_variables(file_spire, file_cygnss)

    for track_id in np.unique(spire['track_ids']):
        try:
            out_dir = os.path.join(dirs['plots'], date_name)
            os.makedirs(out_dir, exist_ok=True)
            file_png = os.path.join(out_dir, str(track_id)+'_'+fname_spire+'.png')
            if os.path.exists(file_png):
                print('file exists!', os.path.basename(file_png))
                continue
            idx = np.where(spire['track_ids'] == track_id)
            if np.floor(spire['pflags'][idx].mean()) != 1:
                continue
            mask = spire['qflags'][idx] != 0
            ant_corr_invalid = np.bitwise_and(spire['qflags'][idx], 128) != 0
            snr_mask = np.bitwise_and(spire['qflags'][idx], 256) != 0
            rfi_mask = np.bitwise_and(spire['qflags'][idx], 512) != 0
            valid = ~snr_mask & ~rfi_mask
            #snr_mask = np.ma.masked_less(spire['snr'][idx], -7).mask
            snr_range = np.max(spire['snr'][idx]) - np.min(spire['snr'][idx])
            rfl_range = np.max(spire['rfl'][idx]) - np.min(spire['rfl'][idx])
            gain_min = np.min([spire['gain'][idx],  cygnss['gain_rt'][idx], cygnss['gain_mean'][idx]])
            gain_max  = np.max([spire['gain'][idx],  cygnss['gain_rt'][idx], cygnss['gain_mean'][idx]])
            inc_min = np.min([spire['inc'][idx],  cygnss['inc_rt'][idx], cygnss['inc_mean'][idx]])
            inc_max  = np.max([spire['inc'][idx],  cygnss['inc_rt'][idx], cygnss['inc_mean'][idx]])


            lon = ((spire['sp_lon'][idx] - 180.0) % 360.0) - 180.0
            lat = spire['sp_lat'][idx]
            extent = [np.min(lon) - 0.5, np.min(lat) - 0.7, np.max(lon) + 0.7, np.max(lat) + 0.5]

            #----------------------------------------------------------
            plt.ioff()
            fig = plt.figure(figsize=(20, 22), constrained_layout=False)
            fig.suptitle('Track:'+str(track_id)+'    '+fname_spire, fontsize=14, y=.95)
            gs = fig.add_gridspec(8, 5)
            #----------------------------------------------------------
            try:
                ax = fig.add_subplot(gs[0, 0])
                _plot_basemap(lon, lat, map_name='shadedrelief') # map_name='etopo', 'bluemarble'

                ax = fig.add_subplot(gs[0, 1:3])
                _plot_basemap(lon, lat, map_name='shadedrelief', extent=extent)

                ax = fig.add_subplot(gs[0, 3:5])
                _plot_lc(lon, lat, extent, dirs, temp_lc_name=os.path.basename(file_cygnss))

            except Exception as e:
                print(e)
                if os.path.exists(os.path.join(dirs['work'], temp_lc_name+'.tif')):
                    os.remove(temp_file)
            #----------------------------------------------------------

            # SNR ----------------------------------------------------------
            ax = fig.add_subplot(gs[1, :])
            ax.set_ylabel('SPIRE'+ '\n' + 'reflect_snr_at_sp (dB)', fontsize=14, color='red')
            #ax.set_ylim(np.min(spire['snr'][idx]), np.min(spire['snr'][idx])+snr_range)
            ax.set_ylim(-10, 20)
            ax.set_xlim(np.min(spire['t'][idx]), np.max(spire['t'][idx]))
            ax.axes.get_xaxis().set_visible(False)
            ax.plot(spire['t'][idx], spire['snr'][idx], '-o', color='black', markersize=1, linewidth=1)
            ax.plot(spire['t'][idx][mask], spire['snr'][idx][mask], '-o', color='orange', markersize=1, linewidth=1)
            ax.plot(spire['t'][idx][ant_corr_invalid], spire['snr'][idx][ant_corr_invalid], '-o', color='brown', markersize=1, linewidth=1)
            ax.plot(spire['t'][idx][rfi_mask], spire['snr'][idx][rfi_mask], '-o', color='magenta', markersize=1, linewidth=1)
            ax.legend(['qflag = 0', 'qflag != 0', 'rx_ant_gain_correction_invalid != 0', 'warn_possible_rfi != 0'], loc="upper left", ncol=2)
            #-----------------
            ax1 = ax.twinx()
            ax1.plot(spire['t'][idx], cygnss['snr_rt'][idx], '-o', color='blue', markersize=2, linewidth=1)
            ax1.legend(['CYGNSS SNR (+-24h)'], loc="upper center")
            ax1.set_ylabel('CYGNSS'+ '\n' + 'Unbiased ddm_snr_mean (dB)', fontsize=14, color='green')
            ax1.plot(spire['t'][idx], cygnss['snr_mean'][idx], '-o', color='green', markersize=1, linewidth=1, label='Mean of CYGNSS SNR')
            #ax1.set_ylim(np.min(cygnss['snr_mean'][idx]), np.min(cygnss['snr_mean'][idx])+snr_range)
            ax1.set_ylim(-10, 20)


            # Reflectivity ----------------------------------------------------------
            ax = fig.add_subplot(gs[2:4, :])
            ax.set_ylabel('SPIRE'+ '\n' + 'reflectivity_at_sp (dB)', fontsize=14, color='red')
            #ax.set_ylim(np.min(spire['rfl'][idx]), np.min(spire['rfl'][idx])+rfl_range)
            ax.set_ylim(-30, 0)
            ax.set_xlim(np.min(spire['t'][idx]), np.max(spire['t'][idx]))
            ax.axes.get_xaxis().set_visible(False)
            ax.plot(spire['t'][idx], spire['rfl'][idx], '-o', color='orange', markersize=1, linewidth=1)
            #ax.plot(np.ma.array(spire['t'][idx], mask=snr_mask), np.ma.array(spire['rfl'][idx], mask=snr_mask),
            #        '-o', color='red', markersize=3, linewidth=1)
            ax.plot(spire['t'][idx][valid], spire['rfl'][idx][valid], '-o', color='red', markersize=1, linewidth=1)
            ax.legend(['all measurements', 'valid measurements'], loc="upper left", fontsize='x-large')
            #-----------------
            ax1 = ax.twinx()
            ax1.plot(spire['t'][idx], cygnss['rfl_rt'][idx], '-o', color='blue', markersize=2, linewidth=1)
            ax1.legend(['CYGNSS RFL (+-24h)'], loc="upper center")
            ax1.set_ylabel('CYGNSS'+ '\n' + 'rfl (dB)', fontsize=14, color='green')
            ax1.plot(spire['t'][idx], cygnss['rfl_mean'][idx], '-o', color='green', markersize=1, linewidth=1, label='Mean of CYGNSS RFL')
            #ax1.set_ylim(np.min(cygnss['rfl_mean'][idx]), np.min(cygnss['rfl_mean'][idx])+rfl_range)
            ax1.set_ylim(-18, 12)


            # scatter plots ----------------------------------------------------------
            x = np.ma.masked_invalid(spire['snr'][idx])
            y = np.ma.masked_invalid(cygnss['snr_mean'][idx])
            sc_mask = x.mask | y.mask
            xy = np.vstack([x[~sc_mask], y[~sc_mask]])
            z = gaussian_kde(xy)(xy)
            ax = fig.add_subplot(gs[4:6, 0:2])
            ax.set_xlim(-10, 20)
            ax.set_ylim(-10, 20)
            ax.set_xlabel('SPIRE SNR (dB)', fontsize=14)
            ax.set_ylabel('CYGNSS SNR (dB)', fontsize=14)
            plt.scatter(x[~sc_mask], y[~sc_mask], c=z, s=5, cmap='jet')
            ax.text(5, 18, "All measurements (quality flags were not applied)", horizontalalignment='center',)
            #-----------------------
            x = np.ma.masked_invalid(spire['rfl'][idx])
            y = np.ma.masked_invalid(cygnss['rfl_mean'][idx])
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)
            ax = fig.add_subplot(gs[4:6, 3:5])
            ax.set_xlim(-30, 0)
            ax.set_ylim(-18, 12)
            ax.set_xlabel('SPIRE RFL (dB)', fontsize=14)
            ax.set_ylabel('CYGNSS RFL (dB)', fontsize=14)
            plt.scatter(x, y, c=z, s=5, cmap='jet')
            ax.text(-15, 10, "All measurements (quality flags were not applied)", horizontalalignment='center',)

            # Gain ----------------------------------------------------------
            ax = fig.add_subplot(gs[6, :])
            ax.set_xlim(np.min(spire['t'][idx]), np.max(spire['t'][idx]))
            ax.set_ylabel('SPIRE'+ '\n' + 'rx_ant_gain_reflect (dB)', fontsize=14, color='black')
            #ax.set_ylim(gain_min, gain_max)
            ax.set_ylim(-5, 15)
            ax.axes.get_xaxis().set_visible(False)
            ax.plot(spire['t'][idx], spire['cgain'][idx], '-o', color='red', markersize=1, linewidth=2)
            ax.legend(['rx_ant_gain_reflect after correction'], loc="upper left", fontsize='large')
            ax.plot(spire['t'][idx], spire['gain'][idx], '-o', color='black', markersize=1, linewidth=2)

            #-----------------
            ax1 = ax.twinx()
            ax1.plot(spire['t'][idx], cygnss['gain_rt'][idx], '-o', color='blue', markersize=2, linewidth=1)
            ax1.legend(['CYGNSS gain (+-24h)'], loc="upper center")
            ax1.set_ylabel('CYGNSS'+ '\n' + 'sp_rx_gain_mean (dB)', fontsize=14, color='green')
            ax1.plot(spire['t'][idx], cygnss['gain_mean'][idx], '-o', color='green', markersize=1, linewidth=1, label='Mean of CYGNSS Ant. Gain')
            #ax1.set_ylim(gain_min, gain_max)
            ax1.set_ylim(-5, 15)


            # Incidence angle ----------------------------------------------------------
            ax = fig.add_subplot(gs[7, :])
            ax.set_xlim(np.min(spire['t'][idx]), np.max(spire['t'][idx]))
            ax.set_ylabel('SPIRE'+ '\n' + 'sp_incidence_angle (dB)', fontsize=14, color='red')
            #ax.set_ylim(inc_min, inc_max)
            ax.set_ylim(0, 80)
            ax.axes.get_xaxis().set_visible(False)
            ax.plot(spire['t'][idx], spire['inc'][idx], '-o', color='red', markersize=1, linewidth=1)
            #-----------------
            ax1 = ax.twinx()
            ax1.plot(spire['t'][idx], cygnss['inc_rt'][idx], '-o', color='blue', markersize=2, linewidth=1, label='CYGNSS Inc. Angle (+-24h)')
            # ax1.legend(loc="upper left")
            ax1.set_ylabel('CYGNSS'+ '\n' + 'sp_inc_angle (dB)', fontsize=14, color='green')
            ax1.plot(spire['t'][idx], cygnss['inc_mean'][idx], '-o', color='green', markersize=1, linewidth=1, label='Mean of CYGNSS Inc. Angle')
            # ax1.set_ylim(inc_min, inc_max)
            ax1.set_ylim(0, 80)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            #fig.tight_layout()
            # plt.show()
            plt.savefig(file_png, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(e)

def _get_dirs():
    dirs={}
    dirs['work'] = r"/home/ubuntu/_working_dir"
    dirs['dpool'] = r"/home/ubuntu/datapool"
    dirs['gdal_path'] = r"/home/ubuntu/miniconda3/envs/en1/bin"
    dirs['lc'] = r"/home/ubuntu/datapool/external/landcover_esa_cci"
    dirs['cygnss'] = os.path.join(dirs['dpool'], "internal", "temp_working_dir", "2020-12-01_spire_cygnss_collocated_tracks", "prod_0.3.7", "data_v3")
    dirs['plots'] = os.path.join(dirs['dpool'], "internal", "temp_working_dir", "2020-12-01_spire_cygnss_collocated_tracks", "prod_0.3.7", "good_matchs")
    #dirs['plots'] = r"/home/ubuntu/_working_dir"
    dirs['spire'] = os.path.join(dirs['dpool'], "internal", "spire_gnssr", "prod-0.3.7", "l1")
    return dirs


def main():
    dirs = _get_dirs()
    files_cygnss = glob.glob(os.path.join(dirs['cygnss'], "*", "*.nc"))

    """ #######################         temp
    dir_search = r"/home/ubuntu/datapool/internal/temp_working_dir/2020-12-01_spire_cygnss_collocated_tracks/prod_v0.2.2/selected_plots/01_good_match"
    sel_plots = glob.glob(os.path.join(dir_search, "*.png"))
    files_cygnss = []
    track_list = []
    for sel_plot in sel_plots:
        fname_plot = os.path.basename(sel_plot)
        fname = fname_plot.replace('_v0.2.2.nc.png', '_v0.3.7.nc')
        fname = 'cygnss-collocated_Spire_'+fname.split('_Spire_')[1]
        file = glob.glob(os.path.join(dirs['cygnss'], "*", fname))
        if len(file) != 0:
            files_cygnss.append(file[0])
            track_list.append(int(fname_plot.split('_')[0]))
    idx_arr=range(len(files_cygnss))
    
    for idx, tr in idx_arr:
            _plot_file(idx, files_cygnss, track_list, dirs)
    
    """ #######################

    mp_num = 8
    if mp_num == 1:
        for file_cygnss in files_cygnss:
            _plot_file(file_cygnss, dirs)
    else:
        prod_file = partial(_plot_file, dirs=dirs)
        p = mp.Pool(processes=mp_num).map(prod_file, files_cygnss)

if __name__ == "__main__":
    main()




