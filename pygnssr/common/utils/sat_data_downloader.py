import subprocess
import click
from datetime import datetime
import os
import glob
import h5py
from pygnssr.common.time.date_to_doys import date_to_doys, date_to_ymd

__author__ = "Vahid Freeman"
__copyright__ = "Copyright 2020, Spire Global"
__credits__ = ["Vahid Freeman"]
__license__ = ""
__version__ = ""
__maintainer__ = "Vahid Freeman"
__email__ = "vahid.freeman@spire.com"
__status__ = "development"


def download_ascat(dataset_name, dataset_version, date, days, dst, check_results=False):
    """
    This program downloads ASCAT Wind data acquired in recent time specified by year and doy lists

    :param dataset_name: Dataset name
    :param dataset_version: Dataset version
    :param date: Date in following format: "%Y-%m-%d %H:%M:%S" .
    :param days: Number of days before the given date to be used as time filter for searching '
                'and downloading the data files.'
    :param dst: (String) destination parent directory
    :param check_results: (Bolean) If set then, the status of the downloaded files will be checked
                        and redownloaded if corrupted.
    """
    # login variables
    user_name = "vfreeman"
    password = "OyO5Tfz54LxN#fqDouz"

    dataset = dataset_name.lower().split('-')[0].lower()
    level = dataset_name.lower().split('-')[1].upper()
    sat_name = dataset_name.lower().split('-')[2].lower()
    version = dataset_version.lower()

    https_path = "https://podaac-tools.jpl.nasa.gov/drive/files/allData/"+ \
                 dataset+"/preview/"+level+"/"+sat_name+"/25km/"

    dst_path = os.path.join(dst, dataset, level, version, sat_name, "25km")

    def cal_wget(year, doy):
        p1 = subprocess.Popen(["/usr/bin/wget",   "-r", "-np",  "-nH", "-c",  "--cut-dirs",  "9",
                               "--user", user_name, "--password", password, "--reject", "index.html*",
                               https_path+str(year)+"/"+str(doy).zfill(3)+"/", "-P", dst_path+"/"+str(year)+"/"])
        p1.communicate()

    def checksum(file):
        """
        Checks whether the downloaded file is healthy

        :param file: downloaded data file to be checked
        NOTE: a corresponding .md5 file should be available in the directory with the same file name+".md5"
        :return: False if the downloaded file is corrupted (different hash) otherwise True
        """
        file_md5 = file + ".md5"
        if os.path.exists(file_md5):
            os.chdir(os.path.dirname(file))
            p1 = subprocess.Popen(["/usr/bin/md5sum", "-c", os.path.basename(file_md5)], stdout=subprocess.PIPE)
            p1.wait()
            out = p1.communicate()[0]
            if 'OK' in str(out):
                return True
            else:
                return False
        else:
            print("The .md5 file does not exists")
            return False

    year_list, doy_list = date_to_doys(date=datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), num_days=days)
    for y, d in zip(year_list, doy_list):
        cal_wget(y, d)

    if check_results:
        for y, d in zip(year_list, doy_list):
            search_path = dst_path + "/" + str(y) + "/" + str(d).zfill(3)
            files = glob.glob(os.path.join(search_path, '*.nc.gz'))
            for f in files:
                if not checksum(f):
                    os.remove(f)
                    f_md5 = f+'.md5'
                    if os.path.exists(f_md5):
                        os.remove(f_md5)
                    print(datetime.now(), "redownloading...!"+os.path.basename(f))
                    cal_wget(y, d)
                else:
                    print("[CHECKSUM OK]", f)


def download_cygnss(dataset_name, dataset_version, date, days, dst, check_results=False):
    """
    This program downloads CYGNSS data acquired in recent time specified by year and doy lists

    :param dataset_name: Dataset name
    :param dataset_version: Dataset version
    :param date: Date in following format: "%Y-%m-%d %H:%M:%S" .
    :param days: Number of days before the given date to be used as time filter for searching '
                'and downloading the data files.'
    :param dst: (String) destination parent directory
    :param check_results: (Bolean) If set then, the status of the downloaded files will be checked
                        and redownloaded if corrupted.
    """
    # login variables
    user_name = "vfreeman"
    password = "OyO5Tfz54LxN#fqDouz"

    dataset = dataset_name.lower().split('-')[0]
    level = dataset_name.lower().split('-')[1].upper()
    version = dataset_version.lower()
    https_path = "https://podaac-tools.jpl.nasa.gov/drive/files/allData/"+\
                 dataset+"/"+level+"/"+version+"/"
    dst_path = os.path.join(dst, dataset, level, version)

    def cal_wget(year, doy):
        p1 = subprocess.Popen(["/usr/bin/wget",   "-r", "-np",  "-nH", "-c",  "--cut-dirs",  "7",
                               "--user", user_name, "--password", password, "--reject", "index.html*",
                               https_path+str(year)+"/"+str(doy).zfill(3)+"/", "-P", dst_path+"/"+str(year)+"/"])
        p1.communicate()

    def checksum(file):
        """
        Checks whether the downloaded file is healthy

        :param file: downloaded data file to be checked
        NOTE: a corresponding .md5 file should be available in the directory with the same file name+".md5"
        :return: False if the downloaded file is corrupted (different hash) otherwise True
        """
        file_md5 = file + ".md5"
        if os.path.exists(file_md5):
            os.chdir(os.path.dirname(file))
            p1 = subprocess.Popen(["/usr/bin/md5sum", "-c", os.path.basename(file_md5)], stdout=subprocess.PIPE)
            p1.wait()
            out = p1.communicate()[0]
            if 'OK' in str(out):
                return True
            else:
                return False
        else:
            print("The .md5 file does not exists")
            return False

    year_list, doy_list = date_to_doys(date=datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), num_days=days)
    for y, d in zip(year_list, doy_list):
        cal_wget(y, d)

    if check_results:
        for y, d in zip(year_list, doy_list):
            search_path = dst_path + "/" + str(y) + "/" + str(d).zfill(3)
            files = glob.glob(os.path.join(search_path, '*.nc'))
            for f in files:
                if not checksum(f):
                    os.remove(f)
                    f_md5 = f+'.md5'
                    if os.path.exists(f_md5):
                        os.remove(f_md5)
                    print(datetime.now(), "redownloading...!"+os.path.basename(f))
                    cal_wget(y, d)
                else:
                    print("[CHECKSUM OK]", f)


def download_smap(dataset_name, dataset_version, date, days, dst, check_results=False):
    """
    This program downloads SMAP L3 9km or 36km data acquired in recent time specified by year and doy lists

    :param dataset_name: Dataset name
    :param dataset_version: Dataset version
    :param date: Date in following format: "%Y-%m-%d %H:%M:%S" .
    :param days: Number of days before the given date to be used as time filter for searching '
                 'and downloading the data files.'
    :param dst: (String) destination parent directory
    :param check_results: (Bolean) If set then, the status of the downloaded files will be checked
                         and redownloaded if corrupted.
    """
    # login variables
    user_name = "vfreeman"
    password = "WidWHiW!7"

    https_path = "https://n5eil01u.ecs.nsidc.org/SMAP/"+dataset_name.upper()+"."+dataset_version.upper()+"/"
    dst_path = os.path.join(dst, 'smap', dataset_name.upper())

    def call_wget(year, month, day):
        p1 = subprocess.Popen(["wget",   "-r", "-np",  "-nH", "-N",  "--cut-dirs",  "2",
                               "--user", user_name, "--password", password, "--reject", "index.html*",
                               "--load-cookies", "/home/ubuntu/.urs_cookies", "--save-cookies", "/home/ubuntu/.urs_cookies", "--keep-session-cookies",
                               "--no-check-certificate", "--auth-no-challenge=on", "-e", "robots=off",
                               https_path+str(year)+"."+str(month).zfill(2)+"."+str(day).zfill(2)+"/", "-P", dst_path+"/"])
        p1.communicate()

    def checksum(file):
        """
        Checks whether the downloaded file is healthy by opening the file

        :param file: downloaded data file to be checked
        :return: False if the downloaded file is corrupted (different hash) otherwise True
        """
        try:
            h5_file = h5py.File(file, mode='r')
            h5_file.close()
            return True
        except OSError as e:
            print("File reading error!..." + file + "\n" + str(e))
            return False

    year_list, month_list, day_list = date_to_ymd(date=datetime.strptime(date, "%Y-%m-%d %H:%M:%S"), num_days=days)
    for y, m, d in zip(year_list, month_list, day_list):
        call_wget(y, m, d)

    if check_results:
        for y, m, d in zip(year_list, month_list, day_list):
            search_path = dst_path + "/" + str(y)+"."+str(m).zfill(2)+"."+str(d).zfill(2)+"/"
            files = glob.glob(os.path.join(search_path, '*.h5'))
            for f in files:
                if not checksum(f):
                    os.remove(f)
                    print(datetime.now(), "redownloading...!"+os.path.basename(f))
                    call_wget(y, m, d)
                else:
                    print("[CHECKSUM OK]", f)


@click.command()
@click.argument('dataset_name', default='CYGNSS-L1', type=str)
@click.option('--dataset_version', default='default', type=str,
              help='If provided, the given data version will be downloaded')
@click.option('--date', default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), type=str,
              help='Date in following format: "%Y-%m-%d %H:%M:%S"  Default is the current date.')
@click.option('--days', default=14, type=int,
              help='Number of days in the past to be used as time filter for searching '
                   'and downloading the data files. Default is 14 days')
@click.option('-dst', default=r"/home/ubuntu/datapool/external", type=str,
              help='Destination directory. Default is the external datapool in "gnssr S3 bucket"')
def main(dataset_name, dataset_version, date, days, dst):
    """
    This program uses wget-call to download satellite data files.

    :param dataset_name: The name of dataset. The supported datafiles are: CYGNSS-L1-V2.1, SPL3SMP_E.003, SPL3SMP.006
    """
    print(datetime.now(), " Data download started from python code ...")

    cygnss_names = ["cygnss-l1"]
    cygnss_versions = ["v2.1", "v3.0"]
    smap_names = ["spl3smp_e", "spl3smp"]
    smap_versions = ["004", "007"]
    ascat_names = ["ascat-l2-metop_a", "ascat-l2-metop_b", "ascat-l2-metop_c"]
    ascat_versions = ['preview']

    if dataset_name.lower() in cygnss_names:
        if dataset_version == 'default':
            dataset_version = cygnss_versions[0]
        if dataset_version.lower() not in cygnss_versions:
            raise ValueError("The Dataset version: " + dataset_version + " is not supported!")
        download_cygnss(dataset_name, dataset_version, date, days, dst, check_results=True)
    elif dataset_name.lower() in smap_names:
        if dataset_version == 'default':
            dataset_version = smap_versions[smap_names.index(dataset_name.lower())]
        if dataset_version.lower() not in smap_versions:
            raise ValueError("The Dataset version: " + dataset_version + " is not supported!")
        download_smap(dataset_name, dataset_version, date, days, dst, check_results=True)
    elif dataset_name.lower() in ascat_names:
        if dataset_version == 'default':
            dataset_version = ascat_versions[0]
        download_ascat(dataset_name, dataset_version, date, days, dst, check_results=True)
    else:
        raise ValueError("The Dataset " + dataset_name + " is not supported!")
    print(datetime.now(), "Data download is finished!")


if __name__ == "__main__":
    main()
