from pygnssr.common.utils.Equi7Grid import Equi7Tile
import numpy as np
import os
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
from collections import OrderedDict
from functools import partial
import multiprocessing as mp

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


class SmapDataCube(object):
    """
    A general DataCube object for
    SMAP L3 Radiometer Global Daily 9km EASE-Grid Soil Moisture Version-3
    # todo this will be extended for other levels of SMAP product

    Attributes
    ----------




    """
    def __init__(self, ftile, level, dir, flag=None):
        """

        :param ftile:
        :param level:
        :param dir:
        :param SMAP_l3_sample_file:
        """
        self.ftile = ftile.upper()
        self.level = str(level).upper()
        self.dir = dir
        # self.fpath = os.path.join(dir, 'SMAP_SPL3SMP_E.003' + self.level +'_' +self.ftile + '.nc')
        self.fpath = os.path.join(dir, 'SMAP_' + self.level +'_' +self.ftile + '.nc')
        # get tile's project specifications and shape
        tile = Equi7Tile(ftile)
        self.spatialreference = tile.projection()
        self.tile_x_size = tile.shape[0]
        self.tile_y_size = tile.shape[1]

        if flag is not None:
            if flag.lower() == 'r':
                self.read()
            elif flag.lower() == 'w':
                self.write()
            else:
                raise ValueError('Flag is not understandable!  Valid flags:"r", "w"')

    def read(self):
        self.nc = Dataset(self.fpath, 'r')

    def write(self):
        if os.path.exists(self.fpath):
            raise ValueError('The output netCDF file exists already ! '+ self.fpath)

        # todo find a proper solution for activating diskless creation of nc. the process is very slow without this option
        ## diskless reading has been activated again,
        ## solution: working always with compressed netCDF files,
        ## copy dst file (already compressed one) to working directory instead of creating a new file and copying variables
        # self.nc = Dataset(self.fpath, 'w')
        self.nc = Dataset(self.fpath, 'w', diskless=True, persist=True)

        self._set_common_attrs()

        if self.level == 'L3':
            self._l3_set_general_attrs()
            self._l3_create_variables()
        else:
            raise ValueError("The only supported SMAP product is level-3")

    def _set_common_attrs(self):
        # set global attributes
        self.nc.history = 'Created/updated on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')
        self.nc.creator = 'SPIRE GLOBAL'
        self.nc.spatial_reference = self.spatialreference
        # define dimensions of the target netCDF
        self.nc.createDimension('x', self.tile_x_size)
        self.nc.createDimension('y', self.tile_y_size)

    def _l3_set_general_attrs(self):
        # set global attributes
        self.nc.description = 'SMAP Level-3 Soil Moisture Data Cube'
        self.nc.source = 'SMAP L3 Radiometer Global Daily 9km EASE-Grid Soil Moisture'
        self.nc.createDimension('sample', None)
        self.nc.createDimension('list', None)
        # todo set data version automatically (at least from file names)
        self.nc.version = '0.4'

    def _l3_create_variables(self):
        v = get_l3_vars_template()

        for p, q in v.items():
            if p == 'processed_files':
                v[p] = self.nc.createVariable(p, q, ('list'))
            else:
                #todo check if fill_values are corrected later
                v[p] = self.nc.createVariable(p, q, ('sample', 'x', 'y'), fill_value=-9999.0)

            if p == 'soil_moisture':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Daily soil moisture composite',
                                                  'coordinates': 'tb_time_utc lat lon',
                                                  'units': 'm3/m3',
                                                  'comment': 'Valid_min: 0.02, Valid_max: Soil porosity'}))
            elif p == 'soil_moisture_error':
                self.nc[p].setncatts(OrderedDict({'long_name': '1-sigma error of the soil_moisture output parameter',
                                                  'coordinates': 'tb_time_utc lat lon',
                                                  'units': 'm3/m3',
                                                  'comment': ''}))
            elif p == 'tb_time_utc':
                self.nc[p].setncatts(OrderedDict({'long_name': 'The average of UTC acquisition time',
                                                  'standard_name': 'time',
                                                  'calendar': 'gregorian',
                                                  'units': 'seconds since 2014-01-01 00:00:00.0',
                                                  'comment': 'The number of seconds since time_coverage_start.'}))
            elif p == 'retrieval_qual_flag':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Data retrieval quality flag',
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
                self.nc[p].setncatts(OrderedDict({'long_name': 'Surface condition flag',
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
            elif p == 'vegetation_water_content':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Daily global composite of the vegetation water content',
                                                  'coordinates': 'tb_time_utc lat lon',
                                                  'units': 'kg/m²',
                                                  'comment': 'This parameter is used as input ancillary data parameter '
                                                             'to the L2_SM_P_E processing software when the baseline'
                                                             'algorithm is used. Valid_min: 0.0  Valid_max: 30.0'}))
            elif p == 'roughness_coefficient':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Daily global composite of roughness coefficient',
                                                  'coordinates': 'tb_time_utc lat lon',
                                                  'units': 'N\A',
                                                  'comment': 'This parameter is the same ‘h’ coefficient'
                                                             'in the ‘tau-omega’ model for a given polarization '
                                                             'channel. Valid_min: 0.0  Valid_max: 3.0'}))
            elif p == 'clay_fraction':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Daily global composite of clay fraction',
                                                  'coordinates': 'tb_time_utc lat lon',
                                                  'units': 'N\A',
                                                  'comment': 'Daily global composite of clay fraction. Valid_min: 0.0'
                                                             ' Valid_max: 1.0'}))
            elif p == 'bulk_density':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Daily global composite of bulk density',
                                                  'coordinates': 'tb_time_utc lat lon',
                                                  'units': 'N\A',
                                                  'comment': 'Daily global composite of bulk density. Valid_min: 0.0 '
                                                             'Valid_max: 3.0'}))
            elif p == 'processed_files':
                self.nc[p].setncatts(OrderedDict({'long_name': 'List of processed files',
                                                    'comment': 'List of successfully processed data files.'}))
            else:
                raise ValueError('Given variable name (read from template is not supported!')

    def close_nc(self):
        if self.nc._isopen:
            self.nc.close()
    #todo# refactor the method's name to make it clear that the file is opened for adaptation
    def open_nc(self):
        self.nc = Dataset(self.fpath, 'r+')


def get_l3_vars_template():
    """
    Selected variables of SMAP level-3 data to be included in SMAP Level-3 data cube
    : return:
    """
    # list of variables
    v = {'soil_moisture': np.float32,
         'soil_moisture_error': np.float32,
         'tb_time_utc': np.float64,
         'retrieval_qual_flag': np.uint16,
         'surface_flag': np.uint16,
         #'vegetation_water_content': np.float32,
         #'roughness_coefficient': np.float32,
         #'clay_fraction': np.float32,
         #'bulk_density': np.float32,
         'processed_files': np.unicode}
    return v
















