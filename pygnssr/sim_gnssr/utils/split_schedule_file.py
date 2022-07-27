import json
import numpy as np
import os
import glob
from multiprocessing import Pool
from datetime import datetime, timedelta
from itertools import repeat
import pyproj


__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2019, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "Development"



def main(dir_in, dir_out, split_num=2):
    """

    """
    files_in =glob.glob(os.path.join(dir_in, '*'))
    for file_in in files_in:
        jtracks = json.load(open(file_in))
        track_num =int(np.ceil(len(jtracks)/split_num))
        for i in range(split_num):
            a=i*track_num
            b=min(a+track_num, len(jtracks)-1)
            jtraks_new=jtracks[a:b]
            file_out = os.path.join(dir_out, os.path.basename(file_in)+'_split_'+str(i+1).zfill(3)+'.json')
            with open(file_out, 'w') as f:
                json.dump(jtraks_new, f, indent="  ")


if __name__ == "__main__":
    dir_in = r"/home/ubuntu/datapool/internal/temp_working_dir/2019-08-19_gnss-r_coverage_maps/schedule_files/MUOS_int_2_hz"
    dir_out = r"/home/ubuntu/datapool/internal/temp_working_dir/2019-08-19_gnss-r_coverage_maps/schedule_files/MUOS_int_2_hz_split"

    main(dir_in, dir_out, split_num=20)



