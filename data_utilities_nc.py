import os
import subprocess
import numpy as np
import random, math
from PIL import Image, ImageFilter
import copy
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import numpy.ma as ma
from plot_utilities import plot_data
import pandas as pd
import datetime as dt


def s3_transfer(local_path, remote_source_name, proc_data_type='file', trans_type= 'upload'):
    if proc_data_type=='file':
        if trans_type== 'upload':
            subprocess.run(['aws', 's3', 'cp', local_path, remote_source_name])
        elif trans_type== 'download':
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            subprocess.run(['aws', 's3', 'cp', remote_source_name, local_path])
    elif proc_data_type=='folder':
        if trans_type== 'upload':
            subprocess.run(['aws', 's3', 'cp', local_path, remote_source_name, '--recursive'])  #the 'recursive command enables iterative file download
        elif trans_type== 'download':
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            subprocess.run(['aws', 's3', 'cp', remote_source_name, local_path, '--recursive'])  #the 'recursive command enables iterative file download
    else:
        raise RuntimeError('download file is unknown;specify if from s3 file or folder') #raise exception; unknown upsampling technique


def read_nc_file(file_path):
    # ---------- Read nc data ----------
    data =  Dataset(os.path.join(file_path), 'r')
    print(data.variables.keys())
    # ---------- Load data ----------
    sm = data.variables['soil_moisture']
    sm_err = data.variables['soil_moisture_error']
    tb_time_utc = data.variables['tb_time_utc']
    retr_flag = data.variables['retrieval_qual_flag']
    surf_flag = data.variables['surface_flag']
    proc_files = data.variables['processed_files']
    
    return [sm, sm_err, tb_time_utc, retr_flag, surf_flag, proc_files]


def reverse_day_rec(data):
    """ function: reverse the records on the first axis of the input
    inp: data to be reversed
    return: reserved arr """
    data_all_init = np.empty([data.shape[1], data.shape[2]])
    data_all = data_all_init[None, :, :] 
    for i in range(data.shape[0]-1, -1, -1):   #reverse reading of day sm
        dt = data[i, :, :]
        data_all = np.concatenate((data_all, dt[None, :, :]), axis=0)   #save cropped pixels/sm

    data_all = data_all[1:, :, :]   #slice out valid data from external run
    return data_all


def retrieve_tm_utc(tm):
    """ function: returns the acquisition time in 'real datetime' format 
    tm: netcdf4 time object """
    tb_time_utc_flat = np.reshape(tm, (tm.shape[0], tm.shape[1]*tm.shape[2]))
    # Get one value per day (pixels in same image have slightly different sensing times)
    dates_smap = np.amax(tb_time_utc_flat, axis=1)
    # Get dates as datetime
    dates_smap_dt = num2date(dates_smap, calendar=tm.calendar, units = tm.units, 
                            only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return dates_smap_dt


def sortDate_keepFill(dates_sm, data_sm, dates_3d, data_qf, fill_val_sm_tm = None,
        qf_fill_val= None, 
        ignored_tm_val=None, skip_inval_data= None):
    """ function: sorts soil moisture measurements and acquisition time without 
    considering the ignored time; returns both sorted soil moisture and time 
    dates_sm: soil moisture aquisition time from netCDF4 object; 1-d array
    data_sm: soil moisture; expected 3-d array 
    dates_3d: soil moisture acquisition times in 3-d array form 
    data_qf: quality flags in 3-d array 
    fill_val_sm_tm: soil moisture fill value for pixels with invalid records
    qf_fiil_val quality flag fill value for pixels with invalid records
    ignored_tm_val: time value for measurements without a single valid soil moisture 
    skip_inval_data: flag that reflects where to drop/skip measurements without a single valid record """ 
    
    h, w = data_sm.shape[1], data_sm.shape[2]
    ignored_sm_tm_val = np.empty((h, w))
    ignored_sm_tm_val[:] = fill_val_sm_tm

    ignored_qf_val = np.empty((h, w))
    ignored_qf_val[:] = qf_fill_val

    dates_sm_cp = dates_sm.copy()  
    data_sm_cp = data_sm.copy()
    dates_3d_cp = dates_3d.copy()
    data_qf_cp = data_qf.copy()

    dates_sm_cp = pd.DataFrame(dates_sm_cp)     #first step to convert time to numpy format
    dates_sm_cp = np.array(dates_sm_cp) #final step to convert time to numpy format
    
    idx_ign_dum = []
    dum_cp = []
    dum_sm = []
    dum_3d_tm = []
    dum_qf = []
    for i in range(dates_sm_cp.shape[0]):
        a = dates_sm_cp[i]
        if a == ignored_tm_val:
            idx_ign_dum += [i]  
        else:
            dum_cp += [dates_sm_cp[i]]
            dum_sm += [data_sm_cp[i]]
            dum_3d_tm += [dates_3d_cp[i]]
            dum_qf += [data_qf_cp[i]]
            
    dum_cp = np.array(dum_cp)
    dum_cp = dum_cp[:, 0]   #convert to row vector
    sort_idx = np.argsort(dum_cp, axis=0)   #get sorting indices
    dum_cp_sort = dum_cp[sort_idx]

    dum_sm = np.array(dum_sm)
    dum_sm_sort = dum_sm[sort_idx, :, :]

    dum_3d_tm = np.array(dum_3d_tm)
    dum_3d_tm_sort = dum_3d_tm[sort_idx, :, :]

    dum_qf = np.array(dum_qf)
    dum_qf_sort = dum_qf[sort_idx, :, :]

    if skip_inval_data == False:
        for j in idx_ign_dum:
            dum_cp_sort = np.insert(dum_cp_sort, j, ignored_tm_val) #dates 1-d array
            dum_sm_sort = np.insert(dum_sm_sort, j, ignored_sm_tm_val, axis=0) #soil moisture 3-d array
            dum_3d_tm_sort = np.insert(dum_3d_tm_sort, j, ignored_sm_tm_val, axis=0) #dates 3-d array
            dum_qf_sort = np.insert(dum_qf_sort, j, ignored_qf_val, axis=0) #quality flag 3-d array
    
    dum_sm_sort = np.reshape(dum_sm_sort, (dum_sm_sort.shape[0], 
                dum_sm_sort.shape[1],  dum_sm_sort.shape[2]))
    
    dum_3d_tm_sort = np.reshape(dum_3d_tm_sort, (dum_3d_tm_sort.shape[0], 
                dum_3d_tm_sort.shape[1],  dum_3d_tm_sort.shape[2]))
    
    dum_qf_sort = np.reshape(dum_qf_sort, (dum_qf_sort.shape[0], 
                dum_qf_sort.shape[1],  dum_qf_sort.shape[2]))

    return dum_cp_sort, dum_sm_sort, dum_3d_tm_sort, dum_qf_sort, sort_idx

def dates_sm_post_process(dates_1d_sm, dates_sm):
    """ function: put the max of all times in a day for all pixels for the same measurement 
    dates_1d_sm: max time for all pixels for all measurements 
    dates_sm: 3-d array with the desired shape that will be filled  """
    dates_sm_new = dates_sm.copy()  #create a dum to be filled
    for i in range(len(dates_1d_sm)):
        dates_sm_new[i, :, :] = dates_1d_sm[i]
    return dates_sm_new

def data_seq_process(data, tm, avg_data = None, exclude_yr = None):
    """ function: sequentializes provided data stack using the provided time stamps
    data: soil moisture in masked format; this is important if daily averaging is employed
    tm: data collection time in netCDF4 format 
    exclude_yr: year to exclude from returned dates; usually the invalid year """
    tb_time_utc_flat = np.reshape(tm, (tm.shape[0], tm.shape[1]*tm.shape[2]))
    # Get one value per day (pixels in same image have slightly different sensing times)
    dates_smap = np.amax(tb_time_utc_flat, axis=1)
    # Get dates as datetime
    dates_smap_dt = num2date(dates_smap, calendar=tm.calendar, units = tm.units, 
                            only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    # dates_smap_dt = [d.date() for d in dates_smap_dt]     #lets us take mainly days into consideration

    # ---------- Create pandas dataframe to correctly average per day (as dates not always consecutive) ----------
    sm_array_fl = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
    df = pd.DataFrame(sm_array_fl, index=dates_smap_dt) #transform to pandas frame, and encode indices as data collection times

    # ---------- average data using the given flag status
    if avg_data == True:
        df = df.groupby(df.index).mean()

    #-----------sort smap using dates, which have been encoded as the indices
    df = df.sort_index(axis=0, ascending=True, inplace=False, na_position='first')  

    # ---------- Recast from dataframe to np array and disregard first nonvalid entry ----------
    sm_array_fl = df.loc[:].values
    sm_day = np.reshape(sm_array_fl, (sm_array_fl.shape[0], data.shape[1], data.shape[2]))

    #-----------sort dates
    dates_smap_dt = pd.DataFrame(dates_smap_dt)
    # dates_smap_dt.sort_values(by=0, inplace=True)

    #-----------extract only valid dates
    if exclude_yr != None:
        dates_smap_dt = pd.to_datetime(dates_smap_dt.iloc[:, 0])    #convert to proper time format
        dates_smap_dt = dates_smap_dt[dates_smap_dt.dt.year != exclude_yr]    #exclude certain year

    dates_smap_dt = np.array(dates_smap_dt) #convert to numpy
    # dates_smap_dt = dates_smap_dt[:, 0]   #converts to row array
    dates_sm_len = dates_smap_dt.shape[0]    #get array length as integer
    sm_day_len = sm_day.shape[0]        #get array length
    sm_day = sm_day[(sm_day_len - dates_sm_len):, :, :]     #slice out array if 'exclude_yr' is set

    return sm_day, dates_smap_dt



def rand_crop_arr(arr, sz):
    """ 
    def: randomly crops arr
    arr: input array to crop
    sz: size of square window used for cropping
    returns: cropped array and corresponding coordinates
 """
    y_max, x_max = arr.shape
    x = np.random.randint(0, x_max - sz)
    y = np.random.randint(0, y_max- sz)         
    return arr[x: x+sz, y: y+sz], np.array([x, x+sz, y, y+sz])

def rand_valid_coord(arr, sz):
    """ def: randomly generates coordinates for random cropping
    arr: input array to crop
    sz: size of square window used for cropping
    returns: valid coordinates for cropping"""

    y_max, x_max = arr[0, :, :].shape
    x = np.random.randint(0, x_max - sz)
    y = np.random.randint(0, y_max- sz) 
    return np.array([x, x+sz, y, y+sz])


def crop_arr(arr, coord):
    """ 
    def: crops arr
    arr: input array to crop
    coord: tuple coordinates of window used for cropping
    returns: cropped array
 """
    x, x_max, y, y_max = coord[0], coord[1], coord[2], coord[3]       
    return arr[x: x_max, y: y_max]

def train_qltyFlag_check(qltyFlag_arr, coord):   
    """ 
    def: checks validity of cropped patch using the corresponding quality flag attributes
    qltyFlag_arr: mask_arr
    coord: four coordinates from which the soil moisture patch was taken
 """
    cropped_mask = qltyFlag_arr[coord[0]: coord[1], coord[2]: coord[3]]
    if np.max(cropped_mask) != 0:    #checks if the max of the quality flag is not equal to 0
        valid_flag = False         #if False, then corresponding croppsed sm patch is invalid 
    else:
        valid_flag = True        #if True, then corresponding croppsed sm patch is valid
    return valid_flag


def inf_train_qltyFlag_check(qltyFlag_arr, coord):
    """ 
    def: during inference, checks validity of cropped patch using the corresponding quality flag attributes
    arr: qltyFlag_arr
    coord: four coordinates from which the soil moisture patch was taken
 """
    cropped_mask = qltyFlag_arr[coord[0]: coord[1], coord[2]: coord[3]]
    if np.max(cropped_mask) != 0:    #checks if the max of the quality flag is greater than 0
        valid_flag = False         #if False, then corresponding croppsed sm patch is invalid 
    else:
        valid_flag = True        #if True, then corresponding croppsed sm patch is valid
    return valid_flag


def process_train_inp(data_agg, qf_arr, tm_agg, crop_sz=None, runs=None):
    """ function randomly crops valid sm patches in the supplied data; the cropped location is consistent across time
    data_agg: data to be randomly cropped to build training data
    qf_arr: quality flag array used for checking valid smap crops
    crop_sz: size of window used for cropping
    runs: number of times to loop through the whole data  """

    data_crop_all_init = np.empty([crop_sz, crop_sz])   #creates an empty
    data_crop_all_run = data_crop_all_init[None, :, :]   #adds an extra dimension; for saving external loop results

    tm_crop_all_init = np.empty([crop_sz, crop_sz])   #creates an empty
    tm_crop_all_run = tm_crop_all_init[None, :, :]   #adds an extra dimension; for saving external loop results

    loc_crop_all_init = np.empty([4])   #creates an empty
    loc_crop_all_run = loc_crop_all_init[None, :]   #adds an extra dimension; for saving external loop results

    for j in range(runs):
        data_all_init = np.empty([crop_sz, crop_sz])    #creates an empty array
        data_crop_all = data_all_init[None, :, :]   #adds an extra dimension; for saving innner loops result

        tm_all_init = np.empty([crop_sz, crop_sz])    #creates an empty array
        tm_crop_all = tm_all_init[None, :, :]   #adds an extra dimension; for saving innner loops result

        loc_all_init = np.empty([4])    #creates an empty array with 4 elements for the cropping coordinates to be stored
        loc_crop_all = loc_all_init[None, :]   #adds an extra dimension; for saving innner loops result

        coord = rand_valid_coord(data_agg, crop_sz)    #generates valid random cropping coordinates for sm

        print('currently processing measurment:', j)
        for i in range(data_agg.shape[0]): #read data 
        # for i in range(data_agg.shape[0]-1, -1, -1): #read data in reversed order, so that it starts from the last day backwards
            # print('frame index being processed:', i, end='\n')
            sm_arr_crop = crop_arr(data_agg[i, :, :], coord)   #gets  cropped soil moisture patch using provided
            tm_arr_crop = crop_arr(tm_agg[i, :, :], coord)   #gets  cropped soil moisture patch using provided
            valid_flag = train_qltyFlag_check(qf_arr[i, :, :], coord)    #test if cropped sm is valid using obtained coordinates

            if valid_flag==True:    #if crooped sm is valid
                data_crop_all = np.concatenate((data_crop_all, sm_arr_crop[None, :, :]), axis=0)   #save cropped sm pixels
                tm_crop_all = np.concatenate((tm_crop_all, tm_arr_crop[None, :, :]), axis=0)   #save cropped times
                loc_crop_all = np.concatenate((loc_crop_all, coord[None, :]), axis=0)   #save cropping coordinates
            else:
                pass    #do nothing; do not append to valid data pool

        data_crop_all = data_crop_all[1:, :, :]   #slice out valid data from internal run
        tm_crop_all = tm_crop_all[1:, :, :]   #slice out valid data from internal run
        loc_crop_all = loc_crop_all[1:, :]   #slice out valid data from internal run
        # print('data_crop_all shape:', data_crop_all.shape)
        # print('data_crop_all_run shape:', data_crop_all_run.shape)
        data_crop_all_run = np.concatenate((data_crop_all_run, data_crop_all), axis=0)
        tm_crop_all_run = np.concatenate((tm_crop_all_run, tm_crop_all), axis=0)
        loc_crop_all_run = np.concatenate((loc_crop_all_run, loc_crop_all), axis=0)
        # print('data_crop_all_run after append shape:', data_crop_all_run.shape)
    data_crop_all_run = data_crop_all_run[1:, :, :]   #slice out valid data from external run
    tm_crop_all_run = tm_crop_all_run[1:, :, :]   #slice out valid data from external run
    loc_crop_all_run = loc_crop_all_run[1:, :]   #slice out valid data from external run
    
    print('finished processing output data')
    return data_crop_all_run, tm_crop_all_run, loc_crop_all_run



def process_train_inp_copy(data_agg, qf_arr, crop_sz=None, runs=None):
    """ function randomly crops valid sm patches in the supplied data; the cropped location is consistent across time
    data_agg: data to be randomly cropped to build training data
    qf_arr: quality flag array used for checking valid smap crops
    crop_sz: size of window used for cropping
    runs: number of times to loop through the whole data  """

    data_crop_all_init = np.empty([crop_sz, crop_sz])   #creates an empty
    data_crop_all_run = data_crop_all_init[None, :, :]   #adds an extra dimension; for saving external loop results
    loc_crop_all_init = np.empty([4])   #creates an empty
    loc_crop_all_run = loc_crop_all_init[None, :]   #adds an extra dimension; for saving external loop results

    for j in range(runs):
        data_all_init = np.empty([crop_sz, crop_sz])    #creates an empty array
        data_crop_all = data_all_init[None, :, :]   #adds an extra dimension; for saving innner loops result

        loc_all_init = np.empty([4])    #creates an empty array with 4 elements for the cropping coordinates to be stored
        loc_crop_all = loc_all_init[None, :]   #adds an extra dimension; for saving innner loops result

        coord = rand_valid_coord(data_agg, crop_sz)    #generates valid random cropping coordinates for sm

        print('currently processing measurment:', j)
        for i in range(data_agg.shape[0]): #read data 
        # for i in range(data_agg.shape[0]-1, -1, -1): #read data in reversed order, so that it starts from the last day backwards
            # print('frame index being processed:', i, end='\n')
            arr_crop = crop_arr(data_agg[i, :, :], coord)   #gets  cropped sm patch using provided
            valid_flag = train_qltyFlag_check(qf_arr[i, :, :], coord)    #test if cropped sm is valid using obtained coordinates

            if valid_flag==True:    #if crooped sm is valid
                data_crop_all = np.concatenate((data_crop_all, arr_crop[None, :, :]), axis=0)   #save cropped pixels/sm
                loc_crop_all = np.concatenate((loc_crop_all, coord[None, :]), axis=0)   #save cropping coordinates
            else:
                pass    #do nothing; do not append to valid data pool

        data_crop_all = data_crop_all[1:, :, :]   #slice out valid data from internal run
        loc_crop_all = loc_crop_all[1:, :]   #slice out valid data from internal run
        # print('data_crop_all shape:', data_crop_all.shape)
        # print('data_crop_all_run shape:', data_crop_all_run.shape)
        data_crop_all_run = np.concatenate((data_crop_all_run, data_crop_all), axis=0)
        loc_crop_all_run = np.concatenate((loc_crop_all_run, loc_crop_all), axis=0)
        # print('data_crop_all_run after append shape:', data_crop_all_run.shape)
    data_crop_all_run = data_crop_all_run[1:, :, :]   #slice out valid data from external run
    loc_crop_all_run = loc_crop_all_run[1:, :]   #slice out valid data from external run
    
    print('finished processing output data')
    return data_crop_all_run, loc_crop_all_run



def add_spatial_gaps(data_ch, mask_sz=None, fill_value= -0.01, nb_insertions=None):
    """ function: simulates spatial gaps for input data for training ML model 
    data_ch: ground-truth data 
    mask: window size for simulation 
    fill_value: value used to denote gap; this has to outside the valid soil moisture range with default of -0.01
    nb_insertions: number of random insertions """ 
    data = data_ch.copy() #performs deep copy so that orignal data is no longer referenced
    y_max, x_max = data[0, :, :].shape #get height and width of any data frame
    mask_sim = np.ones(shape=(mask_sz, mask_sz)) * fill_value #simulate/generate a mask filled with all values of -1 
    # for j in range(nb_insertions):   #controls the number of rounds of insertions
    for i in range(data.shape[0]):
        for j in range(nb_insertions):
            x = np.random.randint(0, x_max - mask_sz)   #randomly generate valid starting x coordinate for masking
            y = np.random.randint(0, y_max - mask_sz)   #randomly generate valid starting y coordinate for masking
            data[i, :, :][x: x + mask_sz, y: y + mask_sz] = mask_sim    #mask insertion; set part input data to values of the mask
    print('finished spatially processing input data with simulated masks')
    return data


def add_temporal_gaps(data_ch, LOOK_BACK=None, fill_value= -0.01, max_insertions=None):
    """ function: simulates temporal gaps for input data for training ML model 
    data_ch: undamaged data/ground-truth data
    LOOK_BACK: time steps from the past to take into account 
    fill_value: value used to denote gap; this has to outside the valid soil moisture range with default of -0.01
    max_insertions: maximum number of insertions """

    data = data_ch.copy()   #performs deep copy so that orignal data is no longer referenced  
    mask_sim = np.ones(shape=(data.shape[1], data.shape[2])) * fill_value #simulate/generate a mask filled with all values of -0.01 
    k = 0
    while k < data.shape[0] - LOOK_BACK: 
        p = np.arange(LOOK_BACK)
        q = np.arange(LOOK_BACK)
        np.random.shuffle(p)    #shuffles array in-place

        x = np.random.randint(1, max_insertions+1) #used for further randomizing the number of mask insertions
        if x> 0: 
            for i in range(x):  #masking performed n times
                data[k + p[i], :, :] = mask_sim
        
        k += LOOK_BACK      #increment pointer to next slice to randomly add gaps/mask
    print('finished temporally processing input data with simulated masks')
    return data

def saveArr2Tiff(arr, iden=None, path=None, clean_path=True):
    """ function loops through data frames, and saves them as tiff images
    arr: data array input
    iden: specify continent and geographic coordinates of file processed
    path: output path for saving images
    clean_path: flag that determines whether to first delete all existing files in the specified path directory """

    if clean_path==True:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
    num = 1
    for i in range(arr.shape[0]):
        dt_arr = arr[i, :, :]
        img = Image.fromarray(dt_arr)
        img.save(path + iden + '_' + str(num)  + '.tif')
        num +=1
    print('finished saving array frames as tiff files')


def empty_folder(mydir):
    filelist = os.listdir(mydir)
    for f in filelist:
        os.remove(os.path.join(mydir, f))


def write_list_to_txtFile(path_write, lst_names, lst):
    '''
    function: write results to a text file
    path_write: the path and file name to the text file to be written into
    lst: the python list to write 
    lst_names: the names attribute to each item in the python list
    write_avg_metric: flag that determines whether you are writing the average results or the per iteration results
    '''

    data = open(path_write, "w")
    for i in range(len(lst)):
        data.write("%s%s\n" % (lst_names[i], lst[i]))
    data.close()



