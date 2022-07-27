import subprocess
import os
import shutil

def compress_netcdf(src_file, dst_file, dir_work=None, copy=False):
    """
    This program does the netcdf compression using nccopy utility and
    moves the compressed file to destination directory

    :param src_file: full path of source file
    :param dst_file: full path of destination file
    :param dir_work: (optional) if provided, it will be used for intermediate file process
    :return:
    """
    if dir_work is None:
        dir_work = os.path.dirname(src_file)

    comp_file = os.path.join(dir_work, os.path.basename(src_file)+'_nccopy_compressed.nc')
    # compress netCDF file
    p = subprocess.Popen(['nccopy', src_file, comp_file, '-d 1 '])
    p.communicate()

    shutil.move(comp_file, dst_file)
    if not copy:
        os.remove(src_file)
