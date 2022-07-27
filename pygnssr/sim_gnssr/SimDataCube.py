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


class SimDataCube(object):
    """

    """
    def __init__(self, ftile, level, dir, flag=None):
        """

        """
        self.ftile = ftile.upper()
        self.level = str(level).upper()
        self.dir = dir
        self.fpath = os.path.join(dir, 'SIM_' + self.level +'_' +self.ftile + '.nc')
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
        self.nc = Dataset(self.fpath, 'w', diskless=True, persist=True)
        self._set_common_attrs()

        if self.level == 'L1':
            self._l1_set_general_attrs()
            self._l1_create_variables()
        else:
            raise ValueError("The only supported SIM product is level-1")

    def _set_common_attrs(self):
        # set global attributes
        self.nc.history = 'Created/updated on: ' + datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S')
        self.nc.creator = 'SPIRE GLOBAL'
        self.nc.spatialrefrence = self.spatialreference
        # define dimensions of the target netCDF
        self.nc.createDimension('x', self.tile_x_size)
        self.nc.createDimension('y', self.tile_y_size)

    def _l1_set_general_attrs(self):
        # set global attributes
        self.nc.description = 'Simulated Level-1 Data Cube'
        self.nc.source = 'Simulated Schedule files'
        self.nc.createDimension('sample', None)
        self.nc.createDimension('list', None)
        # todo set data version automatically (at least from file names)
        self.nc.version = '0.5'

    def _l1_create_variables(self):
        v = get_l1_vars_template()
        for p, q in v.items():
            if p == 'processed_files':
                v[p] = self.nc.createVariable(p, q, ('list'))
            elif p in ['sat_name', 'tx_system']:
                v[p] = self.nc.createVariable(p, q, ('sample', 'x', 'y'), fill_value='NaN', zlib=True, complevel=1)
            else:
                v[p] = self.nc.createVariable(p, q, ('sample', 'x', 'y'), fill_value=-9999.0, zlib=True, complevel=1)
            if p == 'sat_name':
                self.nc[p].setncatts(OrderedDict({'long_name': 'GNSS-R receiver full name'}))
            elif p == 'tx_system':
                self.nc[p].setncatts(OrderedDict({'long_name': 'GNSS Transmitter full name'}))
            elif p == 'sample_time':
                self.nc[p].setncatts(OrderedDict({'long_name': 'The acquisition time',
                                                  'standard_name': 'time',
                                                  'calendar': 'gregorian',
                                                  'units': 'seconds since 2014-01-01 00:00:00.0',
                                                  'comment': 'The number of seconds since time_coverage_start.'}))
            elif p == 'sp_lon':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Specular point longitude'}))
            elif p == 'sp_lat':
                self.nc[p].setncatts(OrderedDict({'long_name': 'Specular point latitude'}))
            elif p == 'rx_ant_gain_reflect':
                self.nc[p].setncatts(OrderedDict({'long_name': 'rx_ant_gain_reflect'}))
            elif p == 'sp_crcg':
                self.nc[p].setncatts(OrderedDict({'long_name': 'sp_crcg'}))
            elif p == 'sp_rcg':
                self.nc[p].setncatts(OrderedDict({'long_name': 'sp_rcg'}))
            elif p == 'sp_elevation_ang':
                self.nc[p].setncatts(OrderedDict({'long_name': 'sp_elevation_ang'}))
            elif p == 'track_id':
                self.nc[p].setncatts(OrderedDict({'long_name': 'track_id'}))
            elif p == 'tx_prn':
                self.nc[p].setncatts(OrderedDict({'long_name': 'tx_prn'}))
            elif p == 'processed_files':
                self.nc[p].setncatts(OrderedDict({'long_name': 'List of processed files',
                                                  'comment': 'List of successfully processed data files.'}))
            else:
                raise ValueError('Given variable name (read from template is not supported!')

    def get_l1_cache_vars(self, sample_size):

        v = get_l1_vars_template()

        names_list = list(v.keys())
        formats_list = list(v.values())
        tp = np.dtype({'names': names_list, 'formats': formats_list})
        cache_arr = np.ma.masked_all((sample_size, self.tile_x_size, self.tile_y_size), dtype=tp)

        return cache_arr

    def close_nc(self):
        if self.nc._isopen:
            self.nc.close()
    #todo# refactor the method's name to make it clear that the file is opened for adaptation
    def open_nc(self):
        self.nc = Dataset(self.fpath, 'r+')


def get_l1_vars_template():
    """
    """
    # list of variables
    v = {'sat_name': np.dtype(('U', 10)),
         'tx_system': np.dtype(('U', 10)),
         'sample_time': np.int32,
         'sp_lon': np.float64,
         'sp_lat': np.float64,
         'rx_ant_gain_reflect': np.float64,
         'sp_crcg':  np.float64,
         'sp_rcg': np.float64,
         'sp_elevation_ang': np.float64,
         'track_id': np.int32,
         'tx_prn': np.int32,
         'processed_files': np.unicode}
    return v

















