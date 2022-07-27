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


class CombSSMDataCube(object):
    """
    A general DataCube object for common attributes of COMB-SSM L2U1 product

    Attributes
    ----------


    """
    def __init__(self, ftile, level, dir, flag=None, overwrite=False):
        """
        :param ftile:
        :param level:
        :param dir:
        :param cygnss_l1_sample_file:
        """
        self.ftile = ftile.upper()
        self.level = str(level).upper()
        self.dir = dir
        self.fpath = os.path.join(dir, 'COMB-SSM_' + self.level +'_' +self.ftile + '.nc')
        # get tile's project specifications and shape
        tile = Equi7Tile(ftile)
        self.spatialreference = tile.projection()
        self.tile_x_size = tile.shape[0]
        self.tile_y_size = tile.shape[1]
        self.overwrite = overwrite

        if flag is not None:
            if flag.lower() == 'r':
                self.read()
            elif flag.lower() == 'w':
                self.write()
            else:
                raise ValueError('Flag is not understandable!  Valid flags:"r", "w"')

    def read(self):
        self.nc = Dataset(self.fpath, 'r')

    def write(self, l1_sample_file=None):
        if os.path.exists(self.fpath):
            if not self.overwrite:
                raise ValueError('The output netCDF file exists already ! ' + self.fpath)
            else:
                # remove old dataset
                os.remove(self.fpath)

        self.nc = Dataset(self.fpath, 'w', diskless=True, persist=True)
        self._set_common_attrs()

        if self.level in ['L2U1', 'L2U2']:
            self._l2u1_set_general_attrs()
            self._l2u1_create_variables()
        else:
            raise ValueError("The supported COMB-SSM products are L2U1, L2U2")

    def _set_common_attrs(self):
        # set global attributes
        self.nc.history = 'Created/updated on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')
        self.nc.creator = 'SPIRE GLOBAL'
        self.nc.spatialrefrence = self.spatialreference
        self.nc.createDimension('x', self.tile_x_size)
        self.nc.createDimension('y', self.tile_y_size)

    def _l2u1_set_general_attrs(self):
        # set global attributes
        self.nc.description = 'SMAP+CYGNSS Combined Surface Soil Moisture Data Cube'
        # todo get the the data version automatically
        self.nc.source = 'SMAP L3 SM (version: 003 and 004) + CYGNSS L1 (version: 2.1)'
        # create groups
        self.cygn_grp = self.nc.createGroup('cygnss')
        self.smap_grp = self.nc.createGroup('smap')
        # define sample dimensions
        self.cygn_grp.createDimension('cygnss_sample', None)
        self.smap_grp.createDimension('smap_sample', None)
        # tot set data version automatically
        self.nc.version = '0.1'

    def _l2u1_create_variables(self):
        v1, v2, v3 = get_l2u1_vars_template()

        for p, q in v1.items():
            v1[p] = self.nc.createVariable(p, q, ('x', 'y'), fill_value=-9999.0)
            if p == 'e7_lon':
                self.nc[p].setncatts(OrderedDict({'long_name': 'equi7 grid point longitude',
                                                  'units': 'degrees',
                                                  'comment': 'longitude of the equi7 grid point.'}))
            elif p == 'e7_lat':
                self.nc[p].setncatts(OrderedDict({'long_name': 'equi7 grid point latitude',
                                                  'units': 'degrees',
                                                  'comment': 'latitude of the equi7 grid point.'}))
            else:
                raise ValueError('Given variable name (read from template is not supported!')

        for p, q in v2.items():
            v2[p] = self.cygn_grp.createVariable(p, q, ('cygnss_sample', 'x', 'y'), fill_value=-9999.0)
            if p == 'time_utc':
                self.cygn_grp[p].setncatts(OrderedDict({'long_name': 'cygnss acquisition time (utc)',
                                                        'calendar': 'gregorian',
                                                        'units': 'seconds since 2014-01-01 00:00:00.0',
                                                        'comment': 'the number of seconds since time coverage start.'}))
            elif p == 'sp_lon':
                self.cygn_grp[p].setncatts(OrderedDict({'long_name': 'cygnss Specular point longitude',
                                                        'units': 'degrees',
                                                        'comment': 'specular point longitude, in degrees, '
                                                                   'at cygnss timestamp_utc'}))
            elif p == 'sp_lat':
                self.cygn_grp[p].setncatts(OrderedDict({'long_name': 'cygnss Specular point latitude',
                                                        'units': 'degrees',
                                                        'comment': 'specular point latitude, in degrees, '
                                                                   'at cygnss timestamp_utc'}))
            elif p == 'rssm':
                self.cygn_grp[p].setncatts(OrderedDict({'long_name': 'relative surface soil moisture',
                                                        'units': '%',
                                                        'comment': 'cygnss uncalibrated relative surface soil moisture '
                                                                   'ranging between 0 and 100.'}))
            elif p == 'cssm':
                self.cygn_grp[p].setncatts(OrderedDict({'long_name': 'Calibrated surface soil moisture',
                                                        'units': 'cm³/cm³',
                                                        'comment': 'cygnss surface soil moisture after calibration with '
                                                                   'auxiliary data in volumetric units.'}))
            elif p == 'cssm_error':
                self.cygn_grp[p].setncatts(OrderedDict({'long_name': 'uncertainty of cygnss cssm',
                                                        'units': 'cm³/cm³',
                                                        'comment': 'reserved variable.'}))
            else:
                raise ValueError('Given variable name (read from template is not supported!')

        for p, q in v3.items():
            v3[p] = self.smap_grp.createVariable(p, q, ('smap_sample', 'x', 'y'), fill_value=-9999.0)
            if p == 'time_utc':
                self.smap_grp[p].setncatts(OrderedDict({'long_name': 'smap acquisition time (utc)',
                                                        'calendar': 'gregorian',
                                                        'units': 'seconds since 2014-01-01 00:00:00.0',
                                                        'comment': 'the number of seconds since '
                                                                   'time coverage start.'}))
            elif p == 'sm':
                self.smap_grp[p].setncatts(OrderedDict({'long_name': 'smap soil moisture',
                                                        'units': 'cm³/cm³',
                                                        'comment': 'Valid_min: 0.02, Valid_max: Soil porosity'}))
            elif p == 'sm_error':
                self.smap_grp[p].setncatts(OrderedDict({'long_name': '1-sigma error of the soil_moisture '
                                                                     'output parameter',
                                                        'units': 'cm³/cm³',
                                                        'comment': 'reserved variable'}))
            elif p == 'retrieval_qual_flag':
                self.smap_grp[p].setncatts(OrderedDict({'long_name': 'Data retrieval quality flag',
                                                        'coordinates': 'tb_time_utc lat lon',
                                                        'comment': '16-bit binary string of 1’s and 0’s that indicate '
                                                                   'whether retrieval was performed or not at a given '
                                                                   'grid cell. When retrieval was performed, it contains '
                                                                   'additional bits to further indicate the exit status '
                                                                   'and quality of the retrieval.',
                                                        'flag_meaning': 'bit-0 Recommended Quality'
                                                                        '  0: Soil moisture retrieval has recommended quality '
                                                                        '  1: Soil moisture retrieval doesn’t have recommended quality'
                                                                        'bit-1 Retrieval Attempted'
                                                                        '  0: Soil moisture retrieval was attempted'
                                                                        '  1: Soil moisture retrieval was skipped'
                                                                        'bit-2 Retrieval Successful'
                                                                        '  0: Soil moisture retrieval was successful'
                                                                        '  1: Soil moisture retrieval was not successful'
                                                                        'bit-3 Retrieval Successful'
                                                                        '  0: Freeze/thaw state retrieval was successful'
                                                                        '  1: Freeze/thaw state retrieval was not successful'
                                                                        'bit-4-15 Undefined 0 (not used in L2_SM_P)'  }))
            elif p == 'surface_flag':
                self.smap_grp[p].setncatts(OrderedDict({'long_name': 'Surface condition flag',
                                                          'coordinates': 'tb_time_utc lat lon',
                                                          'comment': 'Daily global composite of a 16-bit binary string of 1’s '
                                                                     'and 0’s that indicate the presence or absence of certain '
                                                                     'surface conditions at a grid cell. A ‘0’  indicates the '
                                                                     'presence of a surface condition favorable to soil moisture'
                                                                     'retrieval.',
                                                         'flag_meaning': 'bit-0 Static Water (T1=0.05,  T2=0.50):'
                                                                         '  0: Water areal fraction ≤ T1 and IGBP wetland fraction < 0.5: • Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise: • Retrieval skipped for fraction > T2'
                                                                         'bit-1 Radar-derived Water Fraction 0.05 0.50'
                                                                         '  0: Water areal fraction ≤ T1 and IGBP wetland fraction < 0.5: • Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise: • Retrieval skipped for fraction > T2'
                                                                         'bit-2 Coastal Proximity N\A 1.0'
                                                                         '  0: Distance to nearby significant water bodies > T2 (number of 36-km grid cells)'
                                                                         '  1: Otherwise.'
                                                                         'bit-3 Urban Area 0.25 1.00'
                                                                         '  0: Urban areal fraction ≤ T1: Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise: Retrieval skipped for fraction > T2'
                                                                         'bit-4 Precipitation 2.78e-04 	(equivalent to 1.0 mm/hr) 7.06e-03 (equivalent to 25.4 mm/hr)'
                                                                         '  0: Precipitation rate ≤ T1: Retrieval attempted for rate ≤ T2'
                                                                         '  1: Otherwise: Retrieval skipped for rate > T2'
                                                                         'bit-5 Snow 0.05 0.50'
                                                                         '  0: Snow areal fraction ≤ T1: Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise:  Retrieval skipped for fraction > T2'
                                                                         'bit-6 Permanent Ice 0.05 0.50'
                                                                         '  0: Ice areal fraction ≤ T1: Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise: Retrieval skipped for fraction > T2'
                                                                         'bit-7 Frozen Ground (from radiometerderived FT state) 0.05 0.50'
                                                                         '  0: Frozen ground areal fraction ≤ T1: Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise: Retrieval skipped for fraction > T2'
                                                                         'bit-8 Frozen Ground (from modeled effective soil temperature) 0.05 0.50'
                                                                         '  0: Frozen ground areal fraction ≤ T1: Retrieval attempted for fraction ≤ T2'
                                                                         '  1: Otherwise: Retrieval skipped for fraction > T2'
                                                                         'bit-9 Mountainous Terrain 3° 6°'
                                                                         '  0: Slope standard deviation ≤ T1'
                                                                         '  1: Otherwise.'
                                                                         'bit-10 Dense Vegetation 5.0 30.0'
                                                                         '  0: VWC ≤ T1: Retrieval attempted for VWC ≤ T2'
                                                                         '  1: Otherwise: Retrieval skipped for VWC > T2'
                                                                         'bit-11 Nadir Region / Undefined 0 (not used in the product)'
                                                                         'bit-12-15 Undefined 0'}))
            else:
                raise ValueError('Given variable name (read from template is not supported!')

    def close_nc(self):
        if self.nc._isopen:
            self.nc.close()

    #todo# refactor the method's name to make it clear that the file is opened for adaptation
    def open_nc(self):
        self.nc = Dataset(self.fpath, 'r+')


def get_l2u1_vars_template():

    v1 = {'e7_lon': np.float32,
          'e7_lat': np.float32}

    v2 = {'time_utc': np.float64,
          'sp_lon': np.float32,
          'sp_lat': np.float32,
          'rssm': np.float32,
          'cssm': np.float32}
          #'cssm_error': np.float32}

    v3 = {'time_utc': np.float64,
          'sm': np.float32,
          'retrieval_qual_flag': np.uint16,
          'surface_flag': np.uint16}
          #'sm_error': np.float32}

    return v1, v2, v3


def read_comb_ssm_l2u1_dcube(ftile, dir_l2u1):
    # read L1 datacube
    dir_sub_l2u1 = os.path.join(dir_l2u1, ftile.split('_')[0])
    dc_l2u1 = CombSSMDataCube(ftile, 'L2U1', dir_sub_l2u1, flag='r')
    if dc_l2u1.nc is None:
        print('COMB-SSM L2U1 data is not available for ' + ftile + ' ....')
        return None
    return dc_l2u1







