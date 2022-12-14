"""
Note:
    Geos support of Gdal library should be enabled for the accurate spatial
    operation, otherwise, the search overlapped tiles will not be completely
    accurate. Equi7Grid also depends the grids' zone shape files of both aeqd
    and wgs84.

    terminology:
    grid =  grid name (Equi7)
    res= grid resolution/grid spacing (5,10,40,75,500,3000 meters)
    sgrid= subgrid full name (e.g. "AF500M")
    sgrid_id= sub-grid ini (e.g. "AF")
    tile= tile name ("E036N090T6")
    tile_code= tile code representing width of tile (T1:100km, T3:300km,
                T6:600km)
    ftile= full name of tile/sgrid+tile ("AF500M_E036N090T6")
"""

import os
import itertools
import pickle
from osgeo import ogr, osr
import pyproj
import numpy as np
import json

from pygnssr.common.utils import gdalport

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"


def _load_equi7grid_data(module_path):
    # load the data, raise the error if failed to load equi7grid.dat
    equi7_data = None
    fname = os.path.join(os.path.dirname(module_path), "equi7grid.dat")
    with open(fname, "rb") as f:
        equi7_data = pickle.load(f)
    return equi7_data


def get_ftile_names(dir_dpool, grid_res=3000, sgrid_ids=['AS', 'NA', 'SA', 'AF', 'EU', 'OC'], land=True, eq=False):
    """
    returns a list of ftiles (full name of equi7tiles)

    :param dir_dpool: data pool path
    :param grid_res: Equi7 grid spacing
    :param sgrid_id: id of the Equi7 subgrid
    :param land: if set, only the overlapping land tiles will be returned
    :param eq: if set, tiles limited to equtorial orbit will be returned

    :return: list of ftiles
    """
    grid_res = int(grid_res)
    if grid_res >= 3000:
        fname_suffix="T6"
    else:
        raise ValueError("Not implemented yet!")

    if land:
        dir_tile_list = os.path.join(dir_dpool, "internal", "misc", "land_tile_list")
    else:
        raise ValueError("Not implemented yet!")

    if land:
        fname_suffix = fname_suffix + '_LAND'
    if eq:
        fname_suffix = fname_suffix + '_EQUTORIAL'

    sgrid_ids = [sgrid_ids] if not isinstance(sgrid_ids, list) else sgrid_ids
    ftiles = []
    for sgrid_id in sgrid_ids:
        tile_list_file = os.path.join(dir_tile_list, sgrid_id.upper() + "_" + fname_suffix + ".json")
        with open(tile_list_file, "r") as f:
            tile_names = json.load(f)
        ftiles.extend([sgrid_id.upper()+str(grid_res)+"M_"+tile for tile in tile_names])
    return ftiles


class Equi7Grid(object):
    """
    Equi7 Grid

    Parameters
    ----------
    res : float
        The tile resolution
    """

    # static attribute
    _static_equi7_data = _load_equi7grid_data(__file__)
    # sub grid IDs
    _static_sgrid_ids = ["NA", "EU", "AS", "SA", "AF", "OC", "AN"]
    # supported grid spacing(rsolution)
    #todo: include 30000m as well that is factor of 600km adn closer to 36km ease grid
    _static_res = [24000, 12000, 6000, 3000, 1000, 500, 75, 40, 25, 10, 5]
    # supported tile widths(rsolution)
    _static_tilecodes = ["T6", "T3", "T1"]


    def __init__(self, res):
        """
        construct Equi7 grid system.

        """
        self.res = int(res)
        # initializing
        self._initialize()

    def _initialize(self):
        """
        initialization
        """

        # check if the equi7grid.data have been loaded successfully
        if Equi7Grid._static_equi7_data is None:
            self.res = None
            raise ValueError("cannot load Equi7Grid ancillary data!")

        if self.res in [24000, 12000, 6000, 3000, 1000, 500, 75]:
            self._tilecode = "T6"
            self._tile_xspan, self._tile_yspan = (600000, 600000)
        elif self.res in [40]:
            self._tilecode = "T3"
            self._tile_xspan, self._tile_yspan = (300000, 300000)
        elif self.res in [25, 10, 5]:
            self._tilecode = "T1"
            self._tile_xspan, self._tile_yspan = (100000, 100000)
        else:
            invalid_res = self.res
            self.res = None
            msg = "Unsupported resolution {:d}!".format(invalid_res)
            msg += "Supported resolution: {}".format(
                str(Equi7Grid._static_res))
            raise ValueError(msg)

        # keep a reference to the _static_equi7_data
        self._equi7_data = Equi7Grid._static_equi7_data

        # list of sub-grid geometries
        self.sgrid_geom_list = [self.get_sgrid_zone_geom(g) for g in self._static_sgrid_ids]

        # list of sub-grid projection/spatial reference
        self.proj_sgrid_list = self._get_proj_sgrid_list()
        # lat/lon project/spatial reference
        self.proj_latlon = pyproj.Proj("EPSG:4326")
        # define project transformers from latlon2xy and oposite
        self._trans_latlon2xy = {}
        self._trans_xy2latlon = {}
        # create pyproj transformer objects for each sub-grid
        for sgrid_id in self._static_sgrid_ids:
            ## Alternative method -------------------------
            # grid_sr = osr.SpatialReference()
            # grid_sr.ImportFromWkt(self.get_sgrid_projection(sgrid_id))
            # proj_sgrid = pyproj.Proj(grid_sr.ExportToProj4())
            ## Alternative method ^^^^^^^^^^^^^^^^^^^^^^^^^
            proj_sgrid = self.proj_sgrid_list[self._static_sgrid_ids.index(sgrid_id)]

            self._trans_latlon2xy[sgrid_id] = pyproj.Transformer.from_proj(self.proj_latlon, proj_sgrid, always_xy=True)
            self._trans_xy2latlon[sgrid_id] = pyproj.Transformer.from_proj(proj_sgrid, self.proj_latlon, always_xy=True)



    @property
    def span(self):
        return self._tile_xspan

    @property
    def tile_xspan(self):
        return self._tile_xspan

    @property
    def tile_yspan(self):
        return self._tile_yspan

    @property
    def sgrid_ids(self):
        return Equi7Grid._static_sgrid_ids

    @property
    def sgrids(self):
        return [sgrid_id + str(self.res) + 'M' for sgrid_id in Equi7Grid._static_sgrid_ids]

    @property
    def tilecode(self):
        return self._tilecode

    def is_coverland(self, ftile):
        """check if tile covers land

        Parameters
        ----------
        ftile : string
            full tile name
        """
        return Equi7Tile(ftile).covers_land()

    def _get_proj_sgrid_list(self):
        # equi7 subgrid projection/spatial references (proj4) format
        p_sgrid_list = []
        for i in range(len(self._static_sgrid_ids)):
            grid_sr = osr.SpatialReference()
            grid_sr.ImportFromWkt(self.get_sgrid_projection(self._static_sgrid_ids[i]))
            p_sgrid_list.append(pyproj.Proj(grid_sr.ExportToProj4()))
        return p_sgrid_list

    def get_sgrid_zone_geom(self, sgrid_id):
        """return sub-grid extent geometry

        Parameters
        ----------
        sgrid_id : string
            sub-grid id string e.g. EU for Europe

        Return
        ------
        OGRGeomtery
            a geometry representing the extent of given sub-grid

        """
        geom = ogr.CreateGeometryFromWkt(self._equi7_data[sgrid_id]["extent"])
        geo_sr = osr.SpatialReference()
        geo_sr.SetWellKnownGeogCS("EPSG:4326")
        geom.AssignSpatialReference(geo_sr)
        return geom

    def get_sgrid_projection(self, sgrid_id):
        """return sub-grid spatial reference in wkt format

        Parameters
        ----------
        sgrid_id : string
            sub-grid id string e.g. EU for Europe

        Return
        ------
        string
            wkt string representing sub-grid spatial reference

        """
        return self._equi7_data[sgrid_id]["project"]

    def get_sgrid_tiles(self, sgrid_id, tilecode=None):
        """return all the tiles in the given sub grid"""
        if tilecode is None:
            tilecode = self.tilecode

        return list(self._equi7_data[sgrid_id]["coverland"][tilecode])

    def identify_sgrid(self, geom):
        """return the overlapped grid ids"""
        sgrid_ids = [sgrid_id for sgrid_id in Equi7Grid._static_sgrid_ids
                     if geom.Intersects(self.get_sgrid_zone_geom(sgrid_id))]

        return sgrid_ids

    def identify_sgrid_from_lonlat(self, lon=None, lat=None, sgrid_id=None):
        """return the overlapped grid ids.
            The code checks first the geom param. If available then lon lat params are ignored
            :param lon: longitude of the point (should be double precision/float64)
            :param lat: latitude of the point (should be double precision/float64)
            :param sgrid_id: (opional) if given then it will be only checked if lat lon is within that sub-grid
        """
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        if sgrid_id is not None:
            poly = self.get_sgrid_zone_geom(sgrid_id)
            if not poly.Intersects(point):
                sgrid_id = None
        else:
            for id in self._static_sgrid_ids:
                poly = self.get_sgrid_zone_geom(id)
                if poly.Intersects(point):
                    sgrid_id = id
                    break

        return sgrid_id

    def identify_tile(self, sgrid_id, location):
        """Return the tile name."""
        east = int(location[0]/ self._tile_xspan) * self._tile_xspan / 100000
        north = int(location[1]/ self._tile_yspan) * self._tile_yspan / 100000
        return "{}{:03}M_E{:03}N{:03}{}".format(sgrid_id, self.res, int(east), int(north), self.tilecode)

    def get_tile_geotags(self, ftile):
        """
        Return the geotags for given tile used as geoinformation for Geotiff
        """
        tile = Equi7Tile(ftile)
        geotags = {'geotransform': tile.geotransform(),
                   'spatialreference': tile.projection()}
        return geotags

    def lonlat2equi7xy_idx(self, lon, lat):
        """
        vectorize version of _lonlat2equi7xy_idx function
        """
        vfunc = np.vectorize(self._lonlat2equi7xy_idx)
        return vfunc(lon, lat)

    def _lonlat2equi7xy_idx(self, lon, lat):
        """
        Finds corresponding Equi7 point for given lat/lon point

        :param lon: longitude of the point
        :param lat: latitude of the point
        :param sgrid_id: sub-grid id
        :return: full tile name, equi7 x coordinate, equi7 y coordinate, x index in the tile, y index in the tile
        """
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(lon, lat)
        sgrid_id = ''
        for id, poly in zip(self._static_sgrid_ids, self.sgrid_geom_list):
            if poly.Intersects(point):
                sgrid_id = id
                break
        if sgrid_id not in self._static_sgrid_ids:
            ftile = 'X' * 18
            xx = -9999
            yy = -9999
            ix = 255
            iy = 255
        else:
            xx, yy = self._trans_latlon2xy[sgrid_id].transform(lon, lat)
            east = int(int(xx / self._tile_xspan) * self._tile_xspan / 100000)
            north = int(int(yy / self._tile_yspan) * self._tile_yspan / 100000)

            ftile = "{}{:03}M_E{:03}N{:03}{}".format(sgrid_id, self.res, east, north, self.tilecode)
            ix = int(int((xx - east * 100000) / self.res))
            iy = int(int((yy - north * 100000) / self.res))
        return ftile, xx, yy, ix, iy

    def lonlat2equi7xy(self, lon, lat, sgrid_id=None):
        """ convert (lon, lat) to Equi7 (sgrid_id, x, y)

        Parameters
        ----------
        lon: number or numpy.ndarray
            longitude
        lat: number or numpy.ndarray
            latitude
        sgrid_id: string, optional
            sgrid_id if known, otherwise it will be found from the coordinates.
            Without a known sgrid_id this method is much slower since each
            point has to be handled separately.

        Return
        ------
        sgrid_id: string
            subgrid id
        x: like longitude
            projected x coordinate
        y: like latitude
            projected y coordinate

        Raises
        ------
        TypeError
            if lon, lat are numpy arrays but no sgrid_id is given
        """
        if sgrid_id is None:
            vfunc = np.vectorize(self._lonlat2xy_no_sgrid)
            return vfunc(lon, lat)
        else:
            return self._lonlat2xy_sgrid(lon, lat, sgrid_id=sgrid_id)

    def _lonlat2xy_no_sgrid(self, lon, lat):
        """ convert (lon, lat) to Equi7 (sgrid_id, x, y)
        without knowledge of subgrid

        Parameters
        ----------
        lon: number
            longitude
        lat: number
            latitude

        Return
        ------
        sgrid_id: string
            subgrid id
        x: number
            projected x coordinate
        y: number
            projected y coordinate
        """

        # create point geometry
        geo_sr = osr.SpatialReference()
        geo_sr.SetWellKnownGeogCS("EPSG:4326")
        geom_pt = ogr.Geometry(ogr.wkbPoint)
        geom_pt.AddPoint(lon, lat)
        geom_pt.AssignSpatialReference(geo_sr)
        # any valid location should be in only one grid
        sgrid_id = self.identify_sgrid(geom_pt)[0]
        # convert to Equi7
        grid_sr = osr.SpatialReference()
        grid_sr.ImportFromWkt(self.get_sgrid_projection(sgrid_id))
        tx = osr.CoordinateTransformation(geo_sr, grid_sr)
        x, y, _ = tx.TransformPoint(lon, lat)

        return np.full_like(x, sgrid_id, dtype=object), x, y

    def _lonlat2xy_sgrid(self, lon, lat, sgrid_id):
        """ convert (lon, lat) to Equi7 (sgrid_id, x, y)
        with knowledge of sgrid id

        Parameters
        ----------
        lon: number or numpy.ndarray
            longitude
        lat: number or numpy.ndarray
            latitude
        sgrid_id: string
            subgrid id

        Return
        ------
        x: like longitude
            projected x coordinate
        y: like latitude
            projected y coordinate
        """

        x, y = self._trans_latlon2xy[sgrid_id].transform(lon, lat)

        return sgrid_id, x, y

    def equi7xy2lonlat(self, sgrid_id, x, y):
        """ convert Equi7 (sgrid_id, x, y) to (lon, lat)

        """

        vfunc = np.vectorize(self._equi7xy2lonlat)
        return vfunc(sgrid_id, x, y)

    def _equi7xy2lonlat(self, sgrid_id, x, y):
        """ convert Equi7 (sgrid_id, x, y) to (lon, lat)

        Parameters
        ----------
        sgrid_id : string
            the sub-grid id (represent continent) of the coordination,
            should be one of ["NA", "EU", "AS", "SA", "AF", "OC", "AN"]
        x : number or numpy.ndarray
            x coordination
        y : number or numpy.ndarray
            y coordination
        Return
        ------
        lon: number of numpy.ndarray
            longitudes in EPSG 4326
        lat: number of numpy.ndarray
            latitudes in EPSG 4326
        """

        lon, lat = self._trans_xy2latlon[sgrid_id].transform(x, y)
        return lon, lat

    def equi7xy2en(self, sgrid_id, x, y, projstring=None, epsg='4326'):
        """ convert Equi7 (sgrid_id, x, y) to (easting, northing)
        in output projection

        Parameters
        ----------
        sgrid_id : string
            the sub-grid id (represent continent) of the coordination,
            should be one of ["NA", "EU", "AS", "SA", "AF", "OC", "AN"]
        x : number or numpy.ndarray
            x coordination
        y : number or numpy.ndarray
            y coordination
        Return
        ------
        lon: number of numpy.ndarray
            easting in output projection
        lat: number of numpy.ndarray
            northing in output projection
        """
        grid_sr = osr.SpatialReference()
        grid_sr.ImportFromWkt(self.get_sgrid_projection(sgrid_id))
        p_grid = pyproj.Proj(grid_sr.ExportToProj4())
        if projstring is None:
            projection = ':'.join(['EPSG', epsg])
            p_geo = pyproj.Proj(projection)
        else:
            projection = projstring
            p_geo = pyproj.Proj(projection)

        transformer = pyproj.Transformer.from_proj(p_grid, p_geo, always_xy=True)
        eastings, northings = transformer.transform(x, y)
        return eastings, northings

    @staticmethod
    def get_tile_extent(ftile):
        """return the extent of the tile in the terms of [minX,minY,maxX,maxY]
        """
        return Equi7Tile(ftile).extent

    @staticmethod
    def get_tile_geotransform(ftile):
        """
        Return the GDAL geotransform list

        Parameters
        ----------
        ftile : string
            full tile name e.g. EU075M_E048N012T6

        Returns
        -------
        list
            a list contain the geotransfrom elements

        """
        return Equi7Tile(ftile).getransform()

    @staticmethod
    def get_index(dst_ftile, src_ftile, get_px_counts=False):
        """
        Return the index for oversammpling src ftile to dst ftile.

        Parameters
        ----------
        dst_ftile : string
            dst full tile name e.g. AF075M_E066N030T6
        src_ftile : string
            src full tile name e.g. AF500M_E066N030T6
        get_px_counts : bool
            keyword for giving as second return output the number of
            fine pixels in individual coarse pixels

        Return
        ------
        index : numpy array
            The index array with the same size as the dst tilename
        px_counts : numpy array
            The number number of fine pixels per coarse pixel
        """

        dtile = Equi7Tile(dst_ftile)
        stile = Equi7Tile(src_ftile)

        # check if dst tile is a sub tile of src tile
        if dtile.llx < stile.llx or dtile.lly < stile.lly \
           or dtile.llx + dtile.span > stile.llx + stile.span \
           or dtile.lly + dtile.span > stile.lly + stile.span:
            raise ValueError("dst tile should be a sub tile of src tile!")

        index_pattern = {"075T6-500T6": (7, 6, 7),
                         "040T3-500T6": (12, 13),
                         "010T1-500T6": (50,),
                         "040T3-075T6": (2, 2, 2, 1, 2, 2, 2, 2),
                         "010T1-075T6": (7, 8),
                         "010T1-040T3": (4,)
                         }

        index_id = "{:03d}{}-{:03d}{}".format(dtile.res, dtile.tilecode,
                                              stile.res, stile.tilecode)
        if index_id in index_pattern:
            pattern = index_pattern[index_id]
            pattern_sum = int(np.sum(pattern))
        else:
            raise ValueError("Unsupported indexing!")

        # create template
        pattern_tmpl = list()
        for i in range(len(pattern)):
            pattern_tmpl.extend([i] * pattern[i])
        # k number of patterns that dtile takes
        k = dtile.size / pattern_sum + 2
        idx = np.tile(pattern_tmpl, k)
        corr = np.repeat(np.arange(k) * len(pattern), pattern_sum)
        idx += corr

        # make x index
        xoff = (dtile.llx - stile.llx) / dtile.res
        # x_n skip number of patterns
        # x_m start index of dtile in the remaining patterns
        x_n, x_m = xoff / pattern_sum, xoff % pattern_sum
        x_idx = idx + (x_n * len(pattern))
        # shift idx to the correct start point
        x_idx = x_idx[x_m:x_m + dtile.size]

        # make y index
        yoff = (dtile.lly + dtile.span - stile.lly - stile.span) / -dtile.res
        y_n, y_m = yoff / pattern_sum, yoff % pattern_sum
        y_idx = idx + (y_n * len(pattern))
        # shift idx to the correct start point
        y_idx = y_idx[y_m:y_m + dtile.size]

        # create index array
        index = np.empty((dtile.size, dtile.size), dtype=np.uint32)
        for i, v in enumerate(y_idx):
            index[i, :] = x_idx + v * stile.size

        if get_px_counts:
            n_pixels = (np.unique(x_idx,return_counts=True)[1]).astype(np.uint16)
            return index, n_pixels
        else:
            return index

    @staticmethod
    def get_tile_code(res):
        res = int(res)
        tile_code = None
        if res in [24000, 12000, 6000, 3000, 1000, 500, 75]:
            tile_code = "T6"
        elif res in [40]:
            tile_code = "T3"
        elif res in [10, 5]:
            tile_code = "T1"
        else:
            msg = "Error: Given resolution %d is not supported!" % res
            raise ValueError(msg)

        return tile_code

    @staticmethod
    def find_overlapped_tiles(ftile, res):
        """ This function will return the corresponding tile of the
        given tile in the given resolution grid system.

        Parameters
        ----------
        ftile : string
            full tile name e.g. AF075M_E066N030T6
        res : int
            the resolution of the grid system

        Return
        ------
        list
            list of found tiles.
        """
        return Equi7Tile(ftile).find_family_tiles(res=res)

    def search_tiles(self, geom_area=None, extent=None, epsg=4326, sgrid_ids=None, coverland=False, gdal_path=None):
        # TODO The method is not working properly!!!! Needs revisit

        """
        Search the tiles which are intersected by the poly_roi area.

        Parameters
        ----------
        geom_area : geometry
            a polygon or multipolygon geometery object representing the ROI
        extent : list
            It is a polygon representing the rectangle region of interest
            in the format of [xmin, ymin, xmax, ymax].
        epsg : str
            EPSG CODE defining the spatial reference system, in which
            the geometry or extent is given. Default is LatLon (EPSG:4326)
        sgrid_ids : list
            grid ID to specified which continents you want to search. Default
            value is None for searching all continents.

        Returns
        -------
        list
            return a list of  the overlapped tiles' name.
            If not found, return empty list.
        """
        # check input grids
        if sgrid_ids is None:
            sgrid_ids = Equi7Grid._static_sgrid_ids
        else:
            sgrid_ids = [x.upper() for x in sgrid_ids]
            if set(sgrid_ids).issubset(set(Equi7Grid._static_sgrid_ids)):
                sgrid_ids = list(sgrid_ids)
            else:
                raise ValueError("Iextentnvalid agrument: grid must one of [ %s ]." %
                                 " ".join(Equi7Grid._static_sgrid_ids))

        if not geom_area and not extent:
            print("Error: either geom or extent should be given as the ROI.")
            return list()

        # obtain the geometry of ROI
        if not geom_area:
            geom_area = gdalport.extent2polygon(extent)
            geom_sr = osr.SpatialReference()
            geom_sr.ImportFromEPSG(epsg)
            geom_area.AssignSpatialReference(geom_sr)

        # load lat-lon spatial reference as the default
        geo_sr = osr.SpatialReference()
        geo_sr.ImportFromEPSG(4326)

        geom_sr = geom_area.GetSpatialReference()
        if geom_sr is None:
            geom_area.AssignSpatialReference(geo_sr)
        elif not geom_sr.IsSame(geo_sr):
            geom_area.TransformTo(geo_sr)

        # intersect the given grid ids and the overlapped ids
        overlapped_grids = self.identify_sgrid(geom_area)
        sgrid_ids = list(set(sgrid_ids) & set(overlapped_grids))

        # finding tiles
        overlapped_tiles = list()
        for sgrid_id in sgrid_ids:
            overlapped_tiles.extend(
                self.__search_sgrid_tiles(geom_area, sgrid_id, coverland))
        return overlapped_tiles

    def __search_sgrid_tiles(self, geom, sgrid_id, coverland):
        """
        Search the tiles which are overlapping with the given grid.

        Parameters
        ----------
        area_geometry : geometry
            It is a polygon geometry representing the region of interest.
        sgrid_id : string
            sub grid ID to specified which continent you want to search.
            Default value is None for searching all continents.

        Returns
        -------
        list
            Return a list of  the overlapped tiles' name.
            If not found, return empty list.
        """
        # get the intersection of the area of interest and grid zone
        intersect = geom.Intersection(self.get_sgrid_zone_geom(sgrid_id))
        if not intersect:
            return list()
        # The spatial reference need to be set again after intersection
        intersect.AssignSpatialReference(geom.GetSpatialReference())
        # transform the area of interest to the grid coordinate system
        grid_sr = osr.SpatialReference()
        grid_sr.ImportFromWkt(self.get_sgrid_projection(sgrid_id))
        intersect.TransformTo(grid_sr)

        # get envelope of the Geometry and cal the bounding tile of the
        envelope = intersect.GetEnvelope()
        x_min = int(envelope[0]) / self._tile_xspan * self._tile_xspan
        x_max = (int(envelope[1]) / self._tile_xspan + 1) * self._tile_xspan
        y_min = int(envelope[2]) / self._tile_yspan * self._tile_yspan
        y_max = (int(envelope[3]) / self._tile_yspan + 1) * self._tile_yspan

        # make sure x_min and y_min greater or equal 0
        x_min = 0 if x_min < 0 else x_min
        y_min = 0 if y_min < 0 else y_min

        # get overlapped tiles
        overlapped_tiles = list()

        for x, y in itertools.product(list(range(int(x_min), int(x_max), self._tile_xspan)),
                                      list(range(int(y_min), int(y_max), self._tile_yspan))):
            geom_tile = gdalport.extent2polygon((x, y, x + self._tile_xspan,
                                                  y + self._tile_yspan))
            if geom_tile.Intersects(intersect):
                ftile = self.identify_tile(sgrid_id, [x, y])
                if not coverland or self.is_coverland(ftile):
                    overlapped_tiles.append(ftile)

        return overlapped_tiles


class Equi7Tile(object):

    """ Equi7 Tile class

    A tile in the Equi7 grid system.
    """

    def __init__(self, ftile):
        """ fetch information from full tile name
        """
        self.ftile = None
        if not Equi7Tile.is_valid(ftile):
            raise ValueError("invalid full tile name!")
        self.ftile = ftile.upper()
        sgrid, tilename = ftile.split("_")
        self.tilename = tilename
        self.sgrid = sgrid
        self.sgrid_id = sgrid[0:2]
        self.res = int(sgrid[2:-1])
        self.llx = int(tilename[1:4]) * 100000
        self.lly = int(tilename[5:8]) * 100000
        self._xspan = int(tilename[-1]) * 100000
        self._yspan = self._xspan
        self.tilecode = tilename[-2:]
        self._xsize = int(self._xspan / self.res)
        self._ysize = int(self._yspan / self.res)

    @property
    def fullname(self):
        return self.ftile

    @property
    def shortname(self):
        return self.tilename

    @property
    def sub_grid(self):
        return self.sgrid

    @property
    def span(self):
        return self._xspan

    @property
    def size(self):
        return self._xsize

    @property
    def shape(self):
        return (self._ysize, self._xsize)

    @property
    def grid_sr(self):
        if not hasattr(self, "_grid_sr"):
            self._grid_sr = osr.SpatialReference()
            self._grid_sr.ImportFromWkt(self.projection())
        return self._grid_sr

    @property
    def extent(self):
        """return the extent of the tile in the terms of [minX,minY,maxX,maxY]
        """
        return [self.llx, self.lly,
                self.llx + self._xspan, self.lly + self._yspan]

    def projection(self):
        return Equi7Grid._static_equi7_data[self.sgrid_id]["project"]

    def geotransform(self):
        """
        Return the GDAL geotransform list

        Parameters
        ----------
        ftile : string
            full tile name e.g. EU075M_E048N012T6

        Returns
        -------
        list
            a list contain the geotransfrom elements

        """
        geot = [self.llx, self.res, 0,
                self.lly + self._yspan, 0, -self.res]
        return geot

    def get_geometry(self):
        tile_geom=  gdalport.extent2polygon((self.llx, self.lly,
                                             self.llx + self._xspan,
                                             self.lly + self._yspan))
        tile_geom.AssignSpatialReference(self.grid_sr)
        return tile_geom

    def get_tile_geotags(self):
        """
        Return geotags for given tile used as geoinformation for Geotiff
        """
        geotags = {'geotransform': self.geotransform(),
                   'spatialreference': self.projection()}

        return geotags

    def find_family_tiles(self, res=None, tilecode=None):
        """find the family tiles which share the same tile extent but
        with different resoluion and tilecode

        Parameters
        ----------
        ftile : string
            full tile name e.g. AF075M_E066N030T6
        res : int
            the resolution of the grid system
        tilecode : string
            tile code string

        Returns
        -------
        list
            list of found tiles.

        Notes
        -----
        Either the res or tile code should be given.
        But if both are given, the res will be used.
        """
        if res is not None:
            tilecode = Equi7Grid.get_tile_code(res)
        elif tilecode is not None:
            tilecode = tilecode
        else:
            raise ValueError("either res or tilecode should be given!")

        # found family tiles
        family_tiles = list()

        if tilecode >= self.tilecode:
            t_span = int(tilecode[-1]) * 100000
            t_east = (self.llx / t_span) * t_span / 100000
            t_north = (self.lly / t_span) * t_span / 100000
            name = "E{:03d}N{:03d}{}".format(t_east, t_north, tilecode)
            family_tiles.append(name)
        else:
            sub_span = int(tilecode[-1]) * 100000
            n = int(self.span / sub_span)
            for x, y in itertools.product(range(n), range(n)):
                s_east = (self.llx + x * sub_span) / 100000
                s_north = (self.lly + y * sub_span) / 100000
                name = "E{:03d}N{:03d}{}".format(s_east, s_north, tilecode)
                family_tiles.append(name)
        return family_tiles

    def covers_land(self):
        """check if tile covers land"""
        land_tiles = Equi7Grid._static_equi7_data[self.sgrid_id]["coverland"]
        return self.shortname in land_tiles[self.tilecode]

    @staticmethod
    def create(equi7xy=None, lonlat=None):
        raise NotImplementedError()

    @staticmethod
    def is_valid(ftile):
        """check if ftile is a valid tile name"""

        ftile = ftile.upper()

        sgrid, tilename = ftile.split("_")
        # check the constant
        if len(tilename) != 10:
            return False
        if tilename[0] != "E" or tilename[4] != "N":
            return False
        # check variables
        if sgrid[0:2] not in Equi7Grid._static_sgrid_ids:
            return False
        res = int(sgrid[2:-1])
        if res not in Equi7Grid._static_res:
            return False
        _east = int(tilename[1:4])
        _north = int(tilename[5:8])
        if _east < 0 or _north < 0:
            return False
        if tilename[-2:] not in Equi7Grid._static_tilecodes:
            return False
        return True


