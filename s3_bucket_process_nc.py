import os
from re import I
import subprocess
from tkinter import Frame
from netCDF4 import Dataset
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from PIL import Image
import copy
from plot_utilities import plot_data
from data_utilities_nc import read_nc_file, process_train_inp, add_spatial_gaps, add_temporal_gaps, saveArr2Tiff
from data_utilities_nc import data_seq_process, sortDate_keepFill, retrieve_tm_utc, dates_sm_post_process
import pandas as pd
from netCDF4 import Dataset, num2date



data_exist = True      #flag that reflects, if data already exists on current machine, and doesn't have to be read and processed
save_full_data = False  #overwrite/save processed data, if doesn't exist
save_trainData = True   #flag for saving the processed patches for ML model training
download_type='file'        #download type is either 'file' or 'folder'
save_full_processed_smapData_path = '/home/ubuntu/Oyebade/gap_filling/proj_data/full_processedSmap/NA/' #path to save full unprocessed SMAP tile
save_trainData_path = '/home/ubuntu/Oyebade/gap_filling/proj_data/trainData/NA/'    #path to save processed SMAP for training 
full_SmapData_name = 'SMAP_FULL_NA6000M_E078N024T6.npz'          #file name used for saving processed full/original SMAP patches for model testing
patch_trainData_name = 'SMAP_PATCH_NA6000M_E078N024T6.npz'          #file name used for saving processed SMAP patches for model training
# proc_cont = 'NA' #specify which continent to be processed; value is one of string --> AF, AS, EU, NA, OC, SA
path_prcoessing_type= 'file'       #value is 'folder' or 'file'; flag for determing if a folder or file is to be processed
inval_fill_val_sm_tm = -9999.0         #fill value for both invalid soil moisture and time data
inval_fill_val_qltyFlag = 55537         #fill value for invalid quality flag data

if path_prcoessing_type== 'folder':
    remote_source_name = 's3://gnss-datapool/internal/datacube/smap_spl3smp_e/dataset/L3/NA6000M/'  #download SMAP for the whole globe/world
    local_path = '/home/ubuntu/Oyebade/gap_filling/proj_data/SMAP_DATA/L3/NA6000M/'
elif path_prcoessing_type== 'file':
    remote_source_name = 's3://gnss-datapool/internal/datacube/smap_spl3smp_e/dataset/L3/NA6000M/SMAP_L3_NA6000M_E078N024T6.nc'
    local_path = '/home/ubuntu/Oyebade/gap_filling/proj_data/SMAP_DATA/L3/NA6000M/SMAP_L3_NA6000M_E078N024T6.nc'

if data_exist==False:
    if download_type=='file':
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        subprocess.run(['aws', 's3', 'cp', remote_source_name, local_path])
    elif download_type=='folder':
        if not os.path.exists(local_path):
            os.makedirs(local_path)
        subprocess.run(['aws', 's3', 'cp', remote_source_name, local_path, '--recursive'])  #the 'recursive command enables iterative file download
    else:
        raise RuntimeError('download file is unknown;specify if from s3 file or folder') #raise exception; unknown upsampling technique


if path_prcoessing_type== 'file':
    data_read = local_path
elif path_prcoessing_type== 'folder':
    p = 0   #index of file in the folder to process
    list_of_files = os.listdir(local_path)
    data_read = os.path.join(local_path, list_of_files[p])


data = read_nc_file(data_read)

#########soil moisture here
data_sm = data[0]   #read soil moisture data in netcdf4 format
data_sm_masked_arr = data_sm[:] #read soil moisture in masked array format
data_sm_arr_ch = ma.getdata(data_sm_masked_arr)    #get actual arrays
data_mask = data_sm_masked_arr.mask     #get corresponding mask
data_mask_shp = data_mask.shape       #get shape of mask
print('soil moisture shape:', data_sm_arr_ch.shape) #print shape

#########time for collection of soil moisture data here
tm = data[2]   #read soil moisture data in netcdf4 format
tm_masked_arr = tm[:] #read soil moisture in masked array format
tm_arr = ma.getdata(tm_masked_arr)    #get actual arrays
tm_mask = tm_masked_arr.mask     #get corresponding mask
tm_mask_shp = data_mask.shape       #get shape of mask
print('time shape:', tm_arr.shape) #print shape

#########quality flags here
qf = data[3]   #read soil moisture data quality flag in netcdf4 format
qf_masked_arr = qf[:] #read soil moisture in masked array format
qf_arr_ch = ma.getdata(qf_masked_arr)    #get actual arrays; use for testing valid pixels
qf_mask = qf_masked_arr.mask     #get corresponding mask in logical form (i.e. True/False)
df_mask_shp = data_mask.shape       #get shape of mask
print('quality flag shape:', qf_masked_arr.shape) #print shape

#########surface flags
sf = data[4]   #read soil moisture data surface flag in netcdf4 format
sf_masked_arr = sf[:] #read soil moisture in masked array format
sf_arr = ma.getdata(sf_masked_arr)    #get actual arrays; use for testing valid pixels
print('surface flag shape:', sf_arr.shape) #print shape

#########read soil mositure and quality flag array sequentially, and subsequently perform averaging
# data_sm_arr_seq, dates_sm_seq = data_seq_process(data_sm_masked_arr, tm, avg_data = False, exclude_yr = None)   #supply masked soil moisture array
# qf_arr_sq, _ = data_seq_process(qf_arr_ch, tm, avg_data = False, exclude_yr = None)        #supply quality flag actual array; not masked

#####retrieve soil moisture acquisition time in 'real datettime' format
dates_sm_utc = retrieve_tm_utc(tm)

#########sort soil moisture and time, but keeping fill soil moisture and 
#time in their original positions
dates_1d_sm, data_sm_arr, dates_sm_ch, qf_arr, sort_idx = sortDate_keepFill(dates_sm_utc, 
                data_sm_arr_ch, tm_arr, qf_arr_ch, fill_val_sm_tm = inval_fill_val_sm_tm,
                qf_fill_val= inval_fill_val_qltyFlag, 
                ignored_tm_val= np.datetime64('2013-12-31T21:13:21.000000000'),
                skip_inval_data=True)

########apply the sorted dates from 1-d time array to the 3-d time array
dates_sm = dates_sm_post_process(dates_1d_sm, dates_sm_ch)

# qf_arr, _, _ = sortDate_keepFill(qf_arr_ch, dates_sm_utc, 
#                         fill_val_sm = inval_fill_val_qltyFlag)
plot_data(data_sm_arr[0, :, :])
plot_data(qf_arr[0, :, :])


#########replace nan in soil moisture with fill-value of -
# data_sm_arr = np.nan_to_num(data_sm_arr, copy=True, nan= fill_value)

#save both full/original sm input and mask data; also load the same data to verify that all is well
if save_full_data==True:
    np.savez(save_full_processed_smapData_path + full_SmapData_name, full_inp= data_sm_arr, full_mask= data_mask, 
    qlty_flag=qf_arr, surf_flag=sf_arr, time_3d = dates_sm, time_1d = dates_1d_sm) 

test_full_data = np.load(save_full_processed_smapData_path + full_SmapData_name)
full_inp = test_full_data['full_inp']
full_mask = test_full_data['full_mask']
qlty_flag = test_full_data['qlty_flag']
surf_flag = test_full_data['surf_flag']
time_3d = test_full_data['time_3d']
time_1d = test_full_data['time_1d']

data_agg = data_sm_arr

# np.isnan(data_agg).any()

#############create output data without gaps for model training 
# data_out, coord_out = process_train_inp_copy(data_agg, qf_arr, crop_sz=30, runs=50)    #get sequential valid crops from original smap data
sm_out, tm_out, coord_out = process_train_inp(data_agg, qf_arr, dates_sm, crop_sz=30, runs=5000)    #get sequential valid crops from original smap data

#############create input data data with both spatial and temporal gaps for model training
data_inp_sp_gap = add_spatial_gaps(sm_out, mask_sz=10, fill_value= -0.01, nb_insertions=2)  #add spatial gaps
data_inp_sp_tem_gap = add_temporal_gaps(data_inp_sp_gap, LOOK_BACK=7, fill_value= -0.01, max_insertions=3)    #add temporal gaps

############save soil moisture input data, output data, acquisition times and cropped coordinates for model training
if save_trainData==True:
    np.savez(save_trainData_path + patch_trainData_name, inp= data_inp_sp_tem_gap, out= sm_out, time= tm_out, coord = coord_out) #save both input and output data and coordinates

    data_all = np.load(save_trainData_path + patch_trainData_name)
    data_inp = data_all['inp']
    data_out = data_all['out']
    time = data_all['time']
    coord = data_all['coord']
    print('length of data_inp:', len(data_inp))


""" saveArr2Tiff(data_inp, cont= proc_cont, path= '/home/ubuntu/Oyebade/gap_filling/proj_data/trainData/NA/train_inp/')
saveArr2Tiff(data_out, cont= proc_cont, path= '/home/ubuntu/Oyebade/gap_filling/proj_data/trainData/NA/train_out/') """

""" Some notes regarding this solution:
You should install awscli (pip install awscli) and configure it. more info here
If you don't want to override existing files if they weren't changed, you can use sync instead of cp subprocess.run(['aws', 's3', 'sync', remote_folder_name, local_path])
Tested on python 3.6. on earlier versions of python you might need to replace subprocess.run with subprocess.call or os.system
The cli command that is executed by this code is aws s3 cp s3://my-bucket/my-dir . --recursive """

