"""
performing gdal operations
"""


import math
import numpy as np
import os
import subprocess
from osgeo import ogr
from osgeo import osr
from osgeo import gdal, gdal_array


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"



def write_geometry(geom, fname, format="shapefile"):
    """ write an geometry to a vector file.

    parameters
    ----------
    geom : Geometry  (could be in form of a list of object)
        geometry object
    fname : string
        full path of the output file name
    format : string
        format name. currently only shape file is supported
    """
    if not isinstance(geom, list):
        geom=[geom]


    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource(os.path.dirname(fname))
    # take the spatial reference from the first object in the lsit
    srs = geom[0].GetSpatialReference()

    dst_layer = dst_ds.CreateLayer(os.path.basename(fname).split('.')[0], srs=srs)
    fd = ogr.FieldDefn('DN', ogr.OFTInteger)
    dst_layer.CreateField(fd)
    #dst_field = 0

    for i, g in enumerate(geom):
        feature = ogr.Feature(dst_layer.GetLayerDefn())
        feature.SetField("DN", i+1)
        feature.SetGeometry(g)
        dst_layer.CreateFeature(feature)

    feature.Destroy()
    # clean tmp file
    dst_ds.Destroy()
    dst_ds = None
    return


def extent2polygon(extent, wkt=None):
    """create a polygon geometry from extent.

    extent : list
        extent in terms of [xmin, ymin, xmax, ymax]
    wkt : string
        project string in well known text format

    """
    area = [(extent[0], extent[1]), ((extent[0] + extent[2])/2, extent[1]), (extent[2], extent[1]),
            (extent[2], (extent[1] + extent[3])/2),
            (extent[2], extent[3]), ((extent[0] + extent[2])/2, extent[3]), (extent[0], extent[3]),
            (extent[0], (extent[1] + extent[3])/2)]
    edge = ogr.Geometry(ogr.wkbLinearRing)
    [edge.AddPoint_2D(x, y) for x, y in area]
    edge.CloseRings()
    geom_area = ogr.Geometry(ogr.wkbPolygon)
    geom_area.AddGeometry(edge)
    if wkt:
        geo_sr = osr.SpatialReference()
        geo_sr.ImportFromWkt(wkt)
        geom_area.AssignSpatialReference(geo_sr)
    return geom_area


def points2polygon(points, wkt=None, epsg=None):
    '''
    :param points: numpy array
                   sorted point pairs in form of [longitudes, latitudes]
    :param wkt: string
                project string in well known text format
    :param epsg: scaler
                 EPSG code
                 Note: wkt keyword will be ignored if teh epsg is given

    :return: ogr geometry objct
    '''
    edge = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in points:
        edge.AddPoint_2D(x, y)
    edge.CloseRings()
    geom_area = ogr.Geometry(ogr.wkbPolygon)
    geom_area.AddGeometry(edge)

    # add spatial reference
    if (epsg is not None) or (wkt is not None):
        geo_sr = osr.SpatialReference()
        if epsg is not None:
            geo_sr.ImportFromEPSG(epsg)
        else:
            geo_sr.ImportFromWkt(wkt)
        geom_area.AssignSpatialReference(geo_sr)

    return geom_area


def call_gdal_util(util_name, gdal_path=None, src_files=None, dst_file=None,
                   options={}):
    """call gdal utility to run the operation.
        http://www.gdal.org/gdal_utilities.html

    Parameters
    ----------
    util_name : string
        pre-defined name of the utility
        (e.g. "gdal_translate": convert raster data between different formats,
        potentially performing some operations like subsettings, resampling,
        and rescaling pixels in the process.)
    src_files : string
        The source dataset name. It can be either file name,
        URL of data source or subdataset name for multi-dataset files.
    dst_file : string
        The destination file name.
    gdal_path : string
        It the path where your gdal installed. If gpt command can not found by
        the os automatically, you should give this path.
    options : dict
        A dictionary of options. You can find all options at
        http://www.gdal.org/gdal_utilities.html

    """
    # define specific options
    _opt_2b_in_quote = ["-mo", "-co"]

    # prepare the command string
    cmd = []
    if util_name == 'gdal_merge':
        cmd.append('python')
        util_name = util_name+'.py'
    gdal_cmd = os.path.join(gdal_path, util_name) if gdal_path else util_name


    # put gdal_cmd in double quotation
    #cmd.append('"%s"' % gdal_cmd)
    cmd.append(gdal_cmd)

    for k, v in options.items():
        if k in _opt_2b_in_quote:
            if (k == "-mo" or k == "-co") and isinstance(v, (tuple, list)):
                for i in range(len(v)):
                    cmd.append(" ".join((k, '"%s"' % v[i])))
            else:
                cmd.append(" ".join((k, '"%s"' % v)))
        else:
            cmd.append(k)
            #if hasattr(v, "__iter__"):
            #    cmd.append(' '.join(map(str, v)))
            #else:
            cmd.append(str(v))

    # add source files and destination file (in double quotation)
    if src_files is not None:
        if type(src_files) == list:
            src_files_str = " ".join(src_files)
        else:
            src_files_str = '"%s"' % src_files
        cmd.append(src_files_str)

    if util_name == 'gdal_merge.py':
        os.makedirs(os.path.dirname(options['-o']), exist_ok=True)
    else:
        cmd.append('"%s"' % dst_file)
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)

    output = subprocess.check_output(" ".join(cmd), shell=True)

    succeed = _analyse_gdal_output(str(output))
    return succeed, output


def _analyse_gdal_output(output):
    """analyse the output from gpt to find if it executes successfully."""

    # return false if "Error" is found.
    if 'error' in output.lower():
        return False
    # return true if "100%" is found.
    elif '100 - done' in output.lower():
        return True
    # otherwise return false.
    else:
        return False


def read_tiff(src_file, sub_rect=None):
    """
    Parameters
    ----------
    src_file: string (required)
            The full path of source dataset.
    sub_rect : (optional)
        Set this keyword to a four-element array, [Xoffset, Yoffset, width,
        height], that specifies a rectangular region within the input raster
        to extract.

    CAUTION: the returned geotags corresponds to input file and not the
             returned src_arr (in case sub_rect is set)
    """
    src_data = gdal.Open(src_file)
    driver = src_data.GetDriver()
    if driver.ShortName != 'GTiff':
        raise OSError("input file is not a tiff file")

    # Fetch the number of raster bands on this dataset.
    raster_count = src_data.RasterCount
    if raster_count != 1:
        raise OSError("Current version of read_tiff function can only handle \
                        1-band tif files!")

    src_band = src_data.GetRasterBand(1)
    no_data_val = src_band.GetNoDataValue()
    datatype = gdal_array.GDALTypeCodeToNumericTypeCode(src_band.DataType)

    # get parameters
    description = src_data.GetDescription()
    metadata = src_data.GetMetadata()
    # Fetch the affine transformation coefficients.
    geotransform = src_data.GetGeoTransform()
    spatialreference = src_data.GetProjection()
    gcps = src_data.GetGCPs()

    # convert all meta key to lowercase
    metadata_lowercase = dict()
    for k, v in metadata.items():
        metadata_lowercase[k.lower()] = v


    tiff_tags = {'description': description,
                 'metadata': metadata_lowercase,
                 'geotransform': geotransform,
                 'spatialreference': spatialreference,
                 'gcps': gcps,
                 'no_data_val': no_data_val,
                 'datatype': datatype,
                 'blockxsize': src_band.GetBlockSize()[0],
                 'blockysize': src_band.GetBlockSize()[1]}

    if sub_rect is None:
        src_arr = src_data.ReadAsArray()
    else:
        src_arr = src_data.ReadAsArray(sub_rect[0], sub_rect[1], sub_rect[2], sub_rect[3])
    
    src_data = None
    return src_arr, tiff_tags


def write_tiff(dst_file, src_arr=None, x_off=0, y_off=0, shape=None, red=None, green=None, blue=None, tiff_tags=None,
               tilesize=512, ct=None, create_new_file=True):
    """write a 2D numpy array as a single band tiff file with tags.

    Parameters
    ----------
    dst_file : full file path
        The full path of output file.
	src_arr : 2d numpy array
        The input image array to be written.
		It will be ignored if red, green, blue keywords are set.
	x_off: int
        writing offset in x direction (columns) (default 0)
    y_off: int
        writing offset in y direction (rows) (default 0)
    shape :  list/tuple (optional)
        If provided, the number of columns and rows of target image are set based on given shape.
        use case: e.g. if it is desired to write the results on disk stepwise (the final shape of image should be known).
    tiff_tags : dict
        The tags need to be written in the tiff file.
    tilesize : integer
        the tile size of the tiled geotiff, None for the not tiled geotiff.
    ct: gdal colrotable
        if available then the colortable will be attached to geotiff file

	red: (optional)
	green: (optional)
	blue: (optional)
		If you are writing a Palette color image, set these keywords equal to the color table vectors, scaled from 0 to 255.
    """

    _numpy2gdal_dtype = {"bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 3,
                         "uint32": 4, "int32": 5, "float32": 6, "float64": 7,
                         "complex64": 10, "complex128": 11}

    if (red is not None) & (green is not None) & (blue is not None):
        src_arr = red
        # input raster is written as a unique band
        nband = 3
    else:
        nband = 1
        if src_arr is None:
            raise ValueError("src_arr is None!")

    # get gdal data type from numpy data type format, dtype is set according to the src_arr (or red band) dtype
    gdal_dtype = _numpy2gdal_dtype[str(src_arr.dtype)]

    if src_arr.ndim != 2:
        raise OSError(' the input data should have 2-dimensions')

    if shape is not None:
        if len(shape) == 2:
            nrow = shape[0]
            ncol = shape[1]
        else:
            raise ValueError("shape keyword should have 2 elements showing the number of rows and columns of image!")
    else:
        ncol = src_arr.shape[1]
        nrow = src_arr.shape[0]

    # geotiff driver
    opt = ["COMPRESS=LZW"]
    if tilesize:
        tilesize = int(tilesize)
        # make sure the tilesize is exponent of 2
        tilesize = 2 ** int(round(math.log(tilesize, 2)))
        opt.append("TILED=YES")
        opt.append("BLOCKXSIZE={:d}".format(tilesize))
        opt.append("BLOCKYSIZE={:d}".format(tilesize))

    if create_new_file or not os.path.exists(dst_file):
        driver = gdal.GetDriverByName('GTiff')
        dst_data_create = driver.Create(dst_file, ncol, nrow, nband, gdal_dtype, opt)
        dst_data_create = None

    dst_data = gdal.Open(dst_file, gdal.GA_Update)

    # attach tags
    if tiff_tags != None:
        if 'description' in tiff_tags.keys():
            if tiff_tags['description'] != None:
                dst_data.SetDescription(tiff_tags['description'])
        if 'metadata' in tiff_tags.keys():
            if tiff_tags['metadata'] != None:
                dst_data.SetMetadata(tiff_tags['metadata'])
        if 'no_data_val' in tiff_tags.keys():
            if tiff_tags['no_data_val'] != None:
                dst_data.GetRasterBand(1).SetNoDataValue(float(tiff_tags['no_data_val']))
        if 'geotransform' in tiff_tags.keys():
            if tiff_tags['geotransform'] != None:
                dst_data.SetGeoTransform(tiff_tags['geotransform'])
        if 'spatialreference' in tiff_tags.keys():
            if tiff_tags['spatialreference'] != None:
                dst_data.SetProjection(tiff_tags['spatialreference'])
        if 'gcps' in tiff_tags.keys():
            if tiff_tags['gcps'] != None:
                if len(tiff_tags['gcps']) != 0:
                    dst_data.SetGCPs(tiff_tags['gcps'], tiff_tags['spatialreference'])

    # set color table
    if ct is not None:
        dst_data.GetRasterBand(1).SetRasterColorTable(ct)

    dst_data.GetRasterBand(1).WriteArray(src_arr, xoff=x_off, yoff=y_off)
    if nband == 3:
        dst_data.GetRasterBand(2).WriteArray(green)
        dst_data.GetRasterBand(3).WriteArray(blue)

    dst_data.FlushCache()
    dst_data = None


def update_metadata(filename, new_meta, clean=False, colormap_path=None):
    """ updates metadata in image header

    Parameters
    ----------
    filename : string. full path string of input file
    new_meta : new meta data in form of a dictionary of tags and values
    clean: By default, only the given metadata tags will be updated.
           If set True, then all meta data tags will be removed before writing the new ones.
    colormap_path: Text file containing RGB values, if provcovided,
                    a gdal color map will be generated and attached

    Raise
    ------
    IOError
        if fail to open the image file
    """
    dataset = gdal.Open(filename, gdal.GA_Update)
    if dataset is None:
        raise IOError("cannot open %s" % filename)
    if clean:
        dataset.SetMetadata(new_meta)
    else:
        meta = dataset.GetMetadata()
        meta.update(new_meta)
        dataset.SetMetadata(meta)

    if colormap_path is not None:
        ct = gen_gdal_ct(colormap_path)
        dataset.GetRasterBand(1).SetRasterColorTable(ct)

    # close data file
    dataset.FlushCache()
    dataset = None


def gen_gdal_ct(colormap_file):
    with open(colormap_file) as f:
        content = f.readlines()

    ct = gdal.ColorTable()
    for i in range(256):
        rgb = content[i].strip().split()
        ct.SetColorEntry(i, (int(rgb[0]), int(rgb[1]), int(rgb[2])))

    return ct

