from pygnssr.common.utils.Equi7Grid import Equi7Tile
import numpy as np
import os
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
from collections import OrderedDict
import multiprocessing as mp

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


class Spire_gnssrDataCube(object):
    """
    A general DataCube object for common attributes of L1 and L2 products

    Attributes
    ----------


    """
    def __init__(self, ftile, level, dir, flag=None, l1_sample_file=None):
        """
        :param ftile:
        :param level:
        :param dir:
        :param spire_gnssr_l1_sample_file:
        """
        self.ftile = ftile.upper()
        self.level = str(level).upper()
        self.dir = dir
        self.fpath = os.path.join(dir, 'SPIRE_GNSSR_' + self.level +'_' +self.ftile + '.nc')
        # get tile's project specifications and shape
        tile = Equi7Tile(ftile)
        self.spatialreference = tile.projection()
        self.tile_x_size = tile.shape[0]
        self.tile_y_size = tile.shape[1]

        if flag is not None:
            if flag.lower() == 'r':
                self.read()
            elif flag.lower() == 'w':
                self.write(l1_sample_file=l1_sample_file)
            else:
                raise ValueError('Flag is not understandable!  Valid flags:"r", "w"')

    def read(self):
        self.nc = Dataset(self.fpath, 'r')

    def write(self, l1_sample_file=None):
        if os.path.exists(self.fpath):
            raise ValueError('The output netCDF file exists already ! '+ self.fpath)
        if self.level in ['L1', 'L2']:
            if l1_sample_file is None:
                # TODO: improve sample file access
                raise ValueError('L1 netCDF sample file path ("spire_gnssr_l1_sample_file" keyword) is required')
            else:
                self.sample_file = l1_sample_file
                self.nc_sample = Dataset(self.sample_file, 'r')

        self.nc = Dataset(self.fpath, 'w', diskless=True, persist=True)
        self._set_common_attrs()

        if self.level == 'L1':
            self._l1_set_general_attrs()
            self._l1_create_variables()
        elif self.level == 'L2':
            self._l2_set_general_attrs()
            self._l2_create_variables()
        elif self.level == 'L2P':
            self._l2p_set_general_attrs()
            self._l2p_create_variables()

    def _set_common_attrs(self):
        # set global attributes
        self.nc.history = 'Created on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')
        self.nc.spatialrefrence = self.spatialreference
        self.nc.createDimension('x', self.tile_x_size)
        self.nc.createDimension('y', self.tile_y_size)
        # todo: create version automatically from the github hash
        self.nc.version = '0.3'
        # todo: add creator tag (SPIRE GLOBAL)

    def _l1_set_general_attrs(self):
        # set global attributes
        self.nc.description = 'spire_gnssr Level-1 Data Cube'
        # todo: read source data version automatically from input files
        self.nc.source = 'spire_gnssr orbital data files Level-1 version 0.3.1'
        # define sample dimension
        self.nc.createDimension('sample', None)
        self.nc.createDimension('list', None)
        # define DDM dimensions of the target netCDF
        #self.nc.createDimension('delay', 16)
        #self.nc.createDimension('doppler', 23)

    def _l1_create_variables(self):
        v1, v3 = get_l1_vars_template()
        for p, q in v1.items():
            v1[p] = self.nc.createVariable(p, q, ('sample', 'x', 'y'))
            # copy attributes
            self.nc[p].setncatts(self.nc_sample[p].__dict__)
            # change units to hav e a unique starting time for all measurements
            if p == 'sample_time':
                # change epoch to sometime before spire_gnssr satellites' launch
                self.nc[p].units = 'seconds since 2016-01-01 00:00:00.0'
                self.nc[p].calendar = 'gregorian'
        """
        for p, q in v2.items():
            v2[p] = self.nc.createVariable(p, q, ('sample',  'x', 'y','delay', 'doppler'))
            # copy attributes
            self.nc[p].setncatts(self.nc_sample[p].__dict__)
        """

        for p, q in v3.items():
            v1[p] = self.nc.createVariable(p, q, ('list'))
            self.nc[p].setncatts(OrderedDict({'long_name': 'List of processed files',
                                              'comment': 'List of successfully processed data files.'}))

    def get_l1_proc_finits(self, nc_in=None, stime=None, etime=None, num_process=1):
        # todo NOT IMPLEMENTED YET!!!
        """

        :param nc_in: (optional) netCDF file object. if not provided the internal spire_gnssr netCDF data will be analysed
        :return: list of processed spire_gnssr level-1 data files
        (just the first 20 characters e.g. cyg03.ddmi.s20170414)
        """
        if nc_in is None:
            nc_in = self.nc

        #get sapcecraft naumber
        sp_num = nc_in.variables['spacecraft_num'][:, :, :]
        # get the units and calender from nc file
        units = nc_in.variables['sample_time'].units
        calendar = nc_in.variables['sample_time'].calendar

        # mask time invalid time variables
        # TODO it is realized that there are sometimes invalid measurements with 'sp_lon' . variables is masked
        #  but not the 'sample_time' variable.
        nc_time = nc_in.variables['sample_time'][:, :, :]
        timenum_now = date2num(datetime.now(), units, calendar=calendar)
        nc_time = np.ma.masked_greater(nc_time, timenum_now)

        # mask dates outside given period
        if (stime is not None) and (etime is not None):
            stnum = date2num(stime, units, calendar=calendar)
            ennum = date2num(etime, units, calendar=calendar)
            nc_time = np.ma.masked_outside(nc_time, stnum, ennum)

        # return None if no valid measurements found
        if len(nc_time[~nc_time.mask]) == 0:
            return None

        # build and return the unique patterns
        def _get_fname(t, sp, units, calendar):
            return 'cyg' + str(sp).zfill(2) + '.ddmi.s' + np.datetime64(num2date(t, units, calendar=calendar)).astype(datetime).strftime("%Y%m%0d")

        vfunc = np.vectorize(_get_fname, excluded=['units', 'calendar'])
        fnames = vfunc(t=nc_time[~nc_time.mask], sp=sp_num[~nc_time.mask], units=units, calendar=calendar)

        # TODO still very slow!!! test if it becomes faster using multi processing instead of vectorizing
        #p = mp.Pool(num_process)
        #p.starmap(_get_fname, nc_time[~nc_time.mask], sp_num[~nc_time.mask])

        proc_files = set(fnames)
        return proc_files

    def get_l1_cache_vars(self, sample_size): #, delay_size, doppler_size):

        v1, _ = get_l1_vars_template()

        names_list = list(v1.keys())
        formats_list = list(v1.values())
        tp = np.dtype({'names': names_list, 'formats': formats_list})
        cache_arr_1 = np.ma.masked_all((sample_size, self.tile_x_size, self.tile_y_size), dtype=tp)

        """
        names_list = list(v2.keys())
        formats_list = list(v2.values())
        tp = np.dtype({'names': names_list, 'formats': formats_list})
        cache_arr_2 = np.ma.masked_all((sample_size, self.tile_x_size, self.tile_y_size, delay_size, doppler_size), dtype=tp)
        """
        return cache_arr_1 #, cache_arr_2


    def _l2_set_general_attrs(self):
        # set global attributes
        self.nc.description = 'spire_gnssr Level-2 Data Cube'
        self.nc.source = 'spire_gnssr Level-1 Data Cube'
        self.nc.pdb_version = '0.1'
        # define sample dimension
        self.nc.createDimension('sample', None)

    def _l2_create_variables(self):
        v1, v2 = get_l2_vars_template()
        for p, q in v1.items():
            v1[p] = self.nc.createVariable(p, q, ('sample', 'x', 'y'))
            # copy attributes
            self.nc[p].setncatts(self.nc_sample[p].__dict__)
        """
        for p in v2.keys():
            # copy attributes
            at = v2[p].copy()
            v2[p] = self.nc.createVariable(p, at['dtype'], ('sample', 'x', 'y'), fill_value=at['fill_value'])
            self.nc[p].setncatts(at['attrs'])
        """
    def _l2p_set_general_attrs(self):
        # set global attributes
        self.nc.description = 'spire_gnssr Level-2P Data Cube'
        self.nc.source = 'spire_gnssr Level-1 Data Cube'
        self.nc.period = 'YYYY-MM-DD_YYYY-MM-DD'
        # define dimensions for storing 10 percentiles to be used for CDF-matching (calibration purpose)
        self.nc.createDimension('cdf', 10)

    def _l2p_create_variables(self):
        v1, v2 = get_l2p_vars_template()

        for p in v1.keys():
            # copy attributes
            at = v1[p].copy()
            v1[p] = self.nc.createVariable(p, at['dtype'], ('x', 'y'), fill_value=at['fill_value'])
            self.nc[p].setncatts(at['attrs'])

        for p in v2.keys():
            # copy attributes
            at = v2[p].copy()
            v2[p] = self.nc.createVariable(p, at['dtype'], ('cdf', 'x', 'y'), fill_value=at['fill_value'])
            self.nc[p].setncatts(at['attrs'])

    def close_nc(self):
        if self.nc._isopen:
            self.nc.close()
        if hasattr(self, 'sample_file'):
            if self.nc_sample._isopen:
                self.nc_sample.close()

    def open_nc(self):
        self.nc = Dataset(self.fpath, 'r+')


def get_l1_vars_template():
    """
    Selected variables of level-1 data to be included in Spire GNSS-R Level-1 data cube
    :return: Four sets of variables in form of dictionaries of variable names and data types
    """

    # list of variables for each reflection
    v1 = {'sample_time': np.float64,
          'sp_lat': np.float32,
          'sp_lon': np.float32,
          'quality_flags': np.uint64,
          'tx_system': np.int16,
          'antenna_reflect': np.int16,
          'reflect_snr_at_sp': np.float32,
          'reflectivity_at_sp': np.float64,
          'track_id': np.int64,
          'attitude_yaw': np.float32,
          'sp_alt': np.float32,
          'rx_to_sp_antenna_theta': np.float32,
          'rx_to_sp_antenna_phi': np.float32,
          'rx_ant_gain_reflect': np.float32,
          'rx_ant_gain_direct': np.float32,
          'rx_ant_gain_correction': np.float32,
          'sp_incidence_angle': np.float32,
          'tx_svn': np.int16}

    ## List of 2D variables
    #v2 = {'power_reflect': np.float64} #power_reflect(sample_time, delay, doppler)

    v3 = {'processed_files': np.unicode}

    return v1, v3


def get_l2_vars_template():
    """
    spire_gnssr Level-2 data variables
    :return: three sets of variables in form of dictionaries of variable names and attributes
    """
    # list of variables that are common between all 4 reflections
    v1 = { 'spacecraft_num': np.int8,
           'sample_time': np.float64,
           'sp_lon': np.float32,
           'sp_lat': np.float32,
           'prn_code': np.int8,
           'sv_num': np.int32,
           'sp_inc_angle': np.float32,
           'quality_flags': np.int32}

    v2 = { 'rfl': { 'dtype': np.float32,
                    'fill_value': '-9999.0',
                    'attrs': {'long_name': 'reflectivity',
                              'units': 'dB',
                              'comment': 'Reflectivity calculated using the radar equation for coherent '
                                         'measurements'}},

           'nrfl': {'dtype': np.float32,
                    'fill_value': '-9999.0',
                    'attrs': {'long_name': 'normalized reflectivity',
                              'units': 'dB',
                              'comment': 'Normalized reflectivity using the slope of regression lin between '
                                         'refelctivity and incidecen angle after calibration and bias correction'}},

           'rssm': {'dtype': np.float32,
                    'fill_value': '-9999.0',
                    'attrs': {'long_name': 'relative surface soil moisture',
                              'units': '%',
                              'comment': 'uncalibrated relative surface soil moisture ranging between '
                                         '0 and 100'}},

           'rssm2': {'dtype': np.float32,
                     'fill_value': '-9999.0',
                     'attrs': {'long_name': 'relative surface soil moisture',
                               'units': '%',
                               'comment': 'just for aggregation test, calculated using 3km params (not the averaged one 12km)'
                                          '0 and 100'}},

           'cssm': {'dtype': np.float32,
                    'fill_value': '-9999.0',
                    'attrs': {'long_name': 'calibrated surface soil moisture',
                              'units': 'cm³/cm³',
                              'comment': 'surface soil moisture after calibration with auxiliary data in '
                                         'volumetric units'}},
           #todo set the flag bitwise in next release
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


def get_l2p_vars_template():
    """
    spire_gnssr Level-2P  data variables
    :return: one set of variables in form of dictionaries of variable names and attributes
    """

    v1 = {'inc_slope': {'dtype':np.float32,
                        'fill_value': '-9999.0',
                        'attrs': {'long_name': 'incidence angle slope',
                                  'units': 'dB/degree',
                                  'comment': 'slope of regression line between reflectivity and incidence '
                                             'angle'}},

          'inc_intcp': {'dtype': np.float32,
                        'fill_value': '-9999.0',
                        'attrs': {'long_name': 'incidence angle intercept',
                                  'units': 'dB',
                                  'comment': 'intercept of regression line between reflectivity and '
                                             'incidence angle'}},

          'dry': { 'dtype':np.float32,
                   'fill_value': '-9999.0',
                   'attrs': {'long_name': 'dry reference',
                             'units': 'dB',
                             'comment': 'dry reference representing lowest reflectivity measurement corresponding '
                                        'lowest soil moisture condition'}},

          'wet': { 'dtype':np.float32,
                   'fill_value': '-9999.0',
                   'attrs': {'long_name': 'wet reference',
                             'units': 'dB',
                             'comment': 'wet reference representing highest reflectivity measurement corresponding '
                                        'highest soil moisture condition'}}}


    v2 = {'perc_rssm': {'dtype': np.float32,
                        'fill_value': '-9999.0',
                        'attrs': {'long_name': 'RSSM percentiles',
                                  'units': 'unitless',
                                  'comment': 'Percentiles of the source data (RSSM) to be used for'
                                             ' CDF-matching with reference data (SMAP SM)'}},

          'perc_smap': {'dtype': np.float32,
                        'fill_value': '-9999.0',
                        'attrs': {'long_name': 'SMAP SM percentiles',
                                  'units': 'cubic meter/cubic meter',
                                  'comment': 'Percentiles of the reference data (SMAP SM) to be used for'
                                             ' CDF-matching with source data (RSSM)'}}}
    return v1, v2


def read_spire_gnssr_l1_dcube(ftile, dir_l1):
    # read L1 datacube
    dir_sub_l1 = os.path.join(dir_l1, ftile.split('_')[0])
    dc_l1 = Spire_gnssrDataCube(ftile, 'L1', dir_sub_l1, flag='r')
    if dc_l1.nc is None:
        print('L1 data is not available for ' + ftile + ' ....')
        return None
    return dc_l1






