import os
import shutil
import gdal
from pygnssr.common.utils.Equi7Grid import Equi7Tile, Equi7Grid
from pygnssr.common.utils import gdalport
import json

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def warp_to_e7grid(src, dir_work, dir_out, gdal_path, src_name='VAR', ftile_list=None, r_method='near',
                   compress=True, overwrite=False, Tiledtiff=False, colormap_file=None):
    """

    """

    if ftile_list is None:
        raise ValueError("ftile list should be given")

    # resample for each tile sequentially
    for ftile in ftile_list:
        # get tile specifications
        tile = Equi7Tile(ftile)
        temp_out_file =os.path.join(dir_work, src_name.upper()+'_'+tile.fullname+'.tif')

        dir_sub_out = os.path.join(dir_out, ftile.split('_')[0])
        os.makedirs(dir_sub_out, exist_ok=True)

        out_file =os.path.join(dir_sub_out, src_name.upper()+'_'+tile.fullname+'.tif')

        # prepare options for gdalwarp
        options = {'-t_srs': tile.projection(), '-of': 'GTiff',
                   '-r': r_method,
                   '-te': " ".join(map(str, tile.extent)),
                   '-tr': "{} -{}".format(tile.res, tile.res)}

        options["-co"] = list()
        if compress:
            options["-co"].append("COMPRESS=LZW")
        if overwrite:
            options["-overwrite"] = " "
        if Tiledtiff:
            options["-co"].append("TILED=YES")
            options["-co"].append("BLOCKXSIZE=512")
            options["-co"].append("BLOCKYSIZE=512")

        # call gdalwarp for resampling
        succeed, _ = gdalport.call_gdal_util('gdalwarp', src_files=src,
                                             dst_file=temp_out_file, gdal_path=gdal_path,
                                             options=options)

        # set color table
        if colormap_file is not None:
            # read colormap file as gdal ct
            ct = gdalport.gen_gdal_ct(colormap_file)
            # open geotiff file for update
            dst_data = gdal.Open(temp_out_file, gdal.GA_Update)
            dst_data.GetRasterBand(1).SetRasterColorTable(ct)
            dst_data.FlushCache()
            dst_data = None

        # move output to final destination
        shutil.move(temp_out_file, out_file)


def main():

    gdal_path = r"/home/ubuntu/miniconda3/envs/en1/bin"
    dir_dpool = r"/home/ubuntu/datapool"
    dir_work = r"/home/ubuntu/_working_dir"

    res = 1000
    grid = Equi7Grid(res)
    #sgrids = grid._static_sgrid_ids
    sgrids = ['AS', 'NA', 'SA', 'AF', 'EU', 'OC']

    ftiles = []
    for sgrid in sgrids:
        dir_tile_list = os.path.join(dir_dpool, "internal", "misc", "land_tile_list")
        tile_list_file = os.path.join(dir_tile_list, sgrid.upper() + "_T6_LAND.json")
        with open(tile_list_file, "r") as f:
            tile_names = json.load(f)
        ftiles.extend([sgrid.upper()+str(grid.res)+"M_"+tile for tile in tile_names])

    src_name = 'esacci-wb'
    src = os.path.join(dir_dpool, "external", "landcover_esa_cci", "ESACCI-LC-L4-WB-Map-150m-P13Y-2000-v4.0.tif")
    ct_file = os.path.join(dir_dpool, "internal", "misc", "color_tables", "gdal", "ct_esacci_water.ct")
    dir_out = os.path.join(dir_dpool, "internal", "datacube", "wb_esa_cci", "dataset")

    #src_name = 'esacci-lc'
    #src =os.path.join(dir_dpool, "external", "landcover_esa_cci", "ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif")
    #ct_file = None
    #dir_out = os.path.join(dir_dpool, "internal", "datacube", "lc_esa_cci")


    # create the output directory if not exists
    os.makedirs(dir_out, exist_ok=True)


    warp_to_e7grid(src, dir_work, dir_out, gdal_path, src_name=src_name,
                   ftile_list=ftiles, r_method='mode', colormap_file=ct_file)


if __name__ == "__main__":
    main()




