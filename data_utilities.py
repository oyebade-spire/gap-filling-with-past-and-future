import os
import numpy as np
import random, math
from PIL import Image, ImageFilter
from keras.utils.data_utils import Sequence
import copy
# import cv2
# import albumentations as A
# from albumentations import (HorizontalFlip, RandomSizedCrop, RandomResizedCrop, Crop, Blur, 
#         RandomScale, RandomCrop, RandomBrightnessContrast,  Flip, SmallestMaxSize, Resize, 
#         RandomRotate90, Normalize, Compose)
import tensorflow as tf
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from plot_utilities import plotMap_same_scale


from pygnssr.common.utils.Equi7Grid import Equi7Tile
from pygnssr.common.utils.gdalport import write_tiff

import pygnssr2.experimental.generate_stacks

def RandomCrop_param(img, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (Image): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    
    if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
        print("range should be of kind (min, max)")
    
    area = img.shape[0] * img.shape[1]
    test = False
    for attempt in range(10):
        target_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if w <= img.shape[0] and h <= img.shape[1]:
            i = random.randint(0, img.shape[0] - w)
            j = random.randint(0, img.shape[1] - h)
            test = True            
            break

    if test == False:
        # Fallback to central crop
        in_ratio = img.shape[0] / img.shape[1]
        if (in_ratio < min(ratio)):
            w = img.shape[0]
            h = w / min(ratio)
        elif (in_ratio > max(ratio)):
            h = img.shape[1]
            w = h * max(ratio)
        else:  # whole image
            w = img.shape[0]
            h = img.shape[1]
            
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - h) // 2
    i_max = int(i + w)
    j_max = int(j + h)
    j = int(j)
    i = int(i)
    return j, i, j_max, i_max


def process_train(img):
    img = Image.open(img)
    # img = np.array(img)
    img = np.asarray(img)
    return img          

def process_test(img):
    img = Image.open(img)
    # img = np.array(img)
    img = np.asarray(img)
    return img            

class DataSequence_train(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=True):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indices)
                
    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x = [self.x[k] for k in inds]
        batch_y = [self.y[k] for k in inds]  
        
        return np.stack([process_train(file_name) for file_name in batch_x]), np.stack([process_test(file_name) for file_name in batch_y])

 
class DataSequence_test(Sequence):
    def __init__(self, x_set, y_set, batch_size_test):
        self.x = x_set
        self.y = y_set
        self.batch_size = batch_size_test       
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([process_test(file_name) for file_name in batch_x]), np.array([process_test(file_name) for file_name in batch_y])


def get_filenames(dir_name, sort_ls=None):
    # Get list of all files in a given directory sorted by name
    appnd_names = []
    list_of_files = os.listdir(dir_name)
    if sort_ls==True:
        list_of_files = sorted(os.listdir(dir_name))
    for file_name in list_of_files:
        # appnd_names.append(list_of_files)
        appnd_names.append(os.path.join(dir_name, file_name))
    return appnd_names



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
    if np.max(cropped_mask) > 0:    #checks if the max of the quality flag is greater than 0
        valid_flag = False         #if False, then corresponding croppsed sm patch is invalid 
    else:
        valid_flag = True        #if True, then corresponding croppsed sm patch is valid
    return valid_flag


def split_trainTest_data(X, z, y, test_split= 0.1):
    data_len = X.shape[0]
    train_split = 1 - test_split
    x_train = X[0: int(train_split*data_len)+1, :, :, :, :]
    x_test = X[int(train_split*data_len): -1:, :, :, :, :]

    z_train = z[0: int(train_split*data_len)+1, :, :, :, :]
    z_test = z[int(train_split*data_len): -1:, :, :, :, :]

    ####output data
    y_train = y[0: int(train_split*data_len)+1, :, :, :]
    y_test = y[int(train_split*data_len): -1:, :, :, :]

    return (x_train, z_train, y_train), (x_test, z_test, y_test)


# Generated training sequences for use in the model.
def temporalize(in_seq, out_seq, lookback=None):
	X, y = list(), list()
	for i in range(len(in_seq)):
		# find the end of this pattern
		end_ix = i + lookback
		# check if we are beyond the sequence
		if end_ix > len(in_seq)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = in_seq[i:end_ix], out_seq[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def temporalize_sm_tm_fut(in_seq, time_seq, out_seq, lookback=7, max_lk_bck_time=11, lookforward=3, max_lk_fwd_time= 3):
    X, y, z, tm_diff_kp = list(), list(), list(), list()
    for i in range(len(in_seq)):
        # print('i:', i)
        # find the end of this pattern
        end_ix  = i + lookback 
        # check if we are beyond the sequence
        if end_ix + lookforward > len(in_seq)-1:
            break 
        # gather input and output parts of the pattern
        # seq_x, seq_y = in_seq[i:end_ix + 1], out_seq[end_ix]
        seq_x, seq_y = in_seq[i:end_ix + lookforward + 1], out_seq[end_ix]

        tm_his = time_seq[i: end_ix + 1]                      #time array for past measurements
        tm_fut = time_seq[end_ix: end_ix + lookforward + 1]   #time array for future measurements
        tm_all = time_seq[i: end_ix + lookforward + 1]        #extract time that covers both past, current and future
        tm_now = time_seq[end_ix]                             #time for current measurement/image

        lk_bck_diff_chk = np.abs(tm_now - tm_his[0, :, :, :])  #maximum time difference for look back
        lk_bck_diff_chk = np.max(lk_bck_diff_chk)

        lk_fwd_diff_chk = np.abs(tm_fut[-1, :, :, :] - tm_now)  #maximum time difference for the future
        lk_fwd_diff_chk = np.max(lk_fwd_diff_chk)
        # print('diff_chk:', diff_chk)

        if lk_bck_diff_chk <= max_lk_bck_time and lk_fwd_diff_chk <= max_lk_fwd_time:      #check hours
            if seq_x.shape != (lookback + lookforward + 1, 30, 30, 1):
                print('i:', i)
                print('seq_x shape:', seq_x.shape)
                raise RuntimeError('problem with data shape')

            X.append(seq_x)
            y.append(seq_y)

        diff_dum = np.empty([time_seq.shape[1], 
                                time_seq.shape[2]])   #creates an empty
        diff_dum = np.expand_dims(diff_dum, axis=-1)                       
        diff_dum = diff_dum[None, :, :]   #adds an extra dimension; for saving external loop results
        
        if lk_bck_diff_chk <= max_lk_bck_time and lk_fwd_diff_chk <= max_lk_fwd_time:      #check hours
            for j in range(len(tm_all)):
            # for j in range(len(tm_his)-1, -1, -1):
                # print('j:', j)
                # print('tm_now shape:', tm_now.shape)
                # print('tm_his shape:', tm_his.shape)
                # if j == len(tm_his)-1:
                #     k= 7
                # else:
                #     k = j-1
                # diff = np.abs(tm_his[j, :, :, :] - tm_his[k, :, :, :])  #time difference 
                diff = np.abs(tm_now - tm_all[j, :, :, :])  #time difference
                diff_dum = np.concatenate((diff_dum, diff[None, :, :, :]), axis=0)
        
            seq_z = diff_dum[1:, :, :, :]       #slice out valid data
            z.append(seq_z)
            tm_diff_kp.append(lk_bck_diff_chk)
    print('finished')
    print('input shape:', np.array(X).shape)
    print('time shape:', np.array(z).shape)
    print('output shape:', np.array(y).shape)
    X, z, y = np.array(X), np.array(z), np.array(y)
    X = np.around(X, decimals=2)
    z = np.around(z, decimals=2)
    y = np.around(y, decimals=2)
    tm_diff_kp = np.array(tm_diff_kp)
    return X, z, y, tm_diff_kp



def time_diff(time_seq, lk_fwd = None):
    """ function: compute the time difference between current measurement/image and other measurements
    time_seq: time sequence """
    lk_fwd_upd = lk_fwd + 1        #update
    diff_dum = np.empty([time_seq.shape[1], 
                                time_seq.shape[2]])   #creates an empty
    # diff_dum = np.expand_dims(diff_dum, axis=-1)                       
    diff_dum = diff_dum[None, :, :]   #adds an extra dimension; for saving external loop results
    if lk_fwd != None:
        tm_now = time_seq[-lk_fwd_upd, :, :]        #get current time
    else:
        lk_fwd = time_seq[-1, :, :]                 #get current time

    for j in range(len(time_seq)):
        diff = np.abs(tm_now - time_seq[j, :, :])  #time difference 
        diff_dum = np.concatenate((diff_dum, diff[None, :, :]), axis=0)
    seq_z = diff_dum[1:, :, :]       #slice out valid data
    seq_z = np.around(seq_z, decimals=2)
    return seq_z


def temporalize_sm_tm_test_fut(in_seq, time_seq, lookback=None, max_tm_diff=264):
    X, z, tm_diff_kp = list(), list(), list()
    for i in range(len(in_seq)):
        # print('i:', i)
        # find the end of this pattern
        end_ix  = i + lookback 
        # check if we are beyond the sequence
        if end_ix > len(in_seq)-1:
            break 
        # gather input and output parts of the pattern
        seq_x = in_seq[i:end_ix + 1]
        tm_his = time_seq[i:end_ix + 1]
        tm_now = time_seq[end_ix]

        diff_chk = np.abs(tm_now - tm_his[0, :, :, :])  #time difference
        diff_chk = np.max(diff_chk)
        # print('diff_chk:', diff_chk)

        if diff_chk <= max_tm_diff:      #check hours
            X.append(seq_x)

        diff_dum = np.empty([time_seq.shape[1], 
                                time_seq.shape[2]])   #creates an empty
        diff_dum = np.expand_dims(diff_dum, axis=-1)                       
        diff_dum = diff_dum[None, :, :]   #adds an extra dimension; for saving external loop results
        
        if diff_chk <= max_tm_diff:      #check difference in hours betwen extreme measurements/images
            for j in range(len(tm_his)):
            # for j in range(len(tm_his)-1, -1, -1):
                # print('j:', j)
                # print('tm_now shape:', tm_now.shape)
                # print('tm_his shape:', tm_his.shape)

                # if j == len(tm_his)-1:
                #     k= 7
                # else:
                #     k = j-1
                # diff = np.abs(tm_his[j, :, :, :] - tm_his[k, :, :, :])  #time difference 
                diff = np.abs(tm_now - tm_his[j, :, :, :])  #time difference 
                diff_dum = np.concatenate((diff_dum, diff[None, :, :, :]), axis=0)
        
            seq_z = diff_dum[1:, :, :, :]       #slice out valid data
            z.append(seq_z)
            tm_diff_kp.append(diff_chk)
    print('finished')
    print('time shape:', np.array(z).shape)
    X, z = np.array(X), np.array(z)
    X = np.around(X, decimals=2)
    z = np.around(z, decimals=2)
    tm_diff_kp = np.array(tm_diff_kp)
    return X, z, tm_diff_kp


def format_time(tm, fmt_curr = 'secs', fmt_to='hrs'):
    """ function: converts time from one format to another format 
    tm: time array to be converted 
    fmt_curr: current format for time; options are 'secs' or 'nanosecs' 
    fmt_to: time format to convert to; options are 'mins', 'hrs', 'days' """

    if fmt_curr == 'secs':
        fac = 1
    elif fmt_curr == 'nanosecs':
        fac = 10**9
    else:
        raise RuntimeError("unknown current time format specified; options are 'secs' or 'nanosecs'")

    if fmt_to == 'mins':
        tm = tm/ (fac * 60)
    elif fmt_to == 'hrs':
        tm = tm/ (fac * 3600)
    elif fmt_to == 'days':
        tm = tm/ (fac * 3600 * 24)
    else:
        raise RuntimeError("unknown time format specified for conversion; options are 'mins', 'hrs' or 'days'")

    tm = np.around(tm, decimals=2)
    return tm


# def format_time(tm, fmt='hrs'):
#     if fmt == 'mins':
#         tm = tm/ (10**9 * 60)
#     elif fmt == 'hrs':
#         tm = tm/ (10**9 * 3600)
#     elif fmt == 'days':
#         tm = tm/ (10**9 * 3600 * 24)
#     tm = np.around(tm, decimals=2)
#     return tm

def temporalize_now(in_seq, out_seq, lookback=None):
	X, y = list(), list()
	for i in range(len(in_seq)):
		# find the end of this pattern
		end_ix = i + lookback
		# check if we are beyond the sequence
		if end_ix > len(in_seq)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = in_seq[i:end_ix + 1], out_seq[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)


def temporalize_test_now(in_seq, lookback=None):
	X = list()
	for i in range(len(in_seq)):
		# find the end of this pattern
		end_ix = i + lookback
		# check if we are beyond the sequence
		if end_ix > len(in_seq)-1:
			break
		# gather input and output parts of the pattern
		seq_x = in_seq[i:end_ix  + 1]
		X.append(seq_x)
	return np.array(X)


# Generated testing sequences for use in the model.
def temporalize_test(in_seq, lookback=None):
	X = list()
	for i in range(len(in_seq)):
		# find the end of this pattern
		end_ix = i + lookback
		# check if we are beyond the sequence
		if end_ix > len(in_seq)-1:
			break
		# gather input and output parts of the pattern
		seq_x = in_seq[i:end_ix]
		X.append(seq_x)
	return np.array(X)


def temporalize_with_coord(in_seq, coord_seq, out_seq, lookback=None):
	X, z, y, = list(), list(), list()
	for i in range(len(in_seq)):
		# find the end of this pattern
		end_ix = i + lookback
		# check if we are beyond the sequence
		if end_ix > len(in_seq)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_z, seq_y  = in_seq[i:end_ix], coord_seq[i:end_ix], out_seq[end_ix]
		X.append(seq_x)		#input patch
		z.append(seq_z)		#input patch coordinates
		y.append(seq_y)		#output patch
	return np.array(X), np.array(z), np.array(y)


def add_gaps_data(data_ch, qf_arr, data_indx= None, mask_sz=None, trials=100, nb_insertions=None):
    data = copy.deepcopy(data_ch)   #performs deep copy so that orignal data is no longer referenced
    y_max, x_max = data[data_indx, :, :].shape #get height and width of any data frame
    mask_sim = np.ones(shape=(mask_sz, mask_sz)) * -1 #simulate/generate a mask filled with all values of -1 
    # mask_sim = np.zeros(shape=(mask_sz, mask_sz)) #simulate/generate a mask filled with all values of zeros
    
    for j in range(nb_insertions):   #controls the number of rounds of insertions
        coord = rand_valid_coord(data, mask_sz) #get random valid coordinates for cropping
        arr_crop = crop_arr(data[data_indx, :, :], coord)   #gets  cropped sm patch using provided
        valid_flag = train_qltyFlag_check(qf_arr[data_indx, :, :], coord)    #test if cropped sm is valid using obtained coordinates
        k = trials
        if valid_flag== True:
            data[data_indx, :, :][coord[0]: coord[1], coord[2]: coord[3]] = mask_sim    #set part input data to values of all -1
        else: 
            while not valid_flag and  k> 0: #loop until valid area in data for masking is found or trial times is exhausted
                arr_crop = crop_arr(data[data_indx, :, :], coord)   #gets  cropped sm patch using provided
                k -= 1  #decrease counter
                if k== 0:
                    raise RuntimeError('Could not find valid area in data for mask insertion')
                else:
                    data[data_indx, :, :][coord[0]: coord[1], coord[2]: coord[3]] = mask_sim     #set part input data to values of all -1
    
    print('finished processing input data with simulated masks')
    return data


def mae_rmse_single_day(data_ground, data_pred, qlty_flg_temp_slc):
    """ function: computes the mae and rmse for a single day using supplied ground truth, prediction and quality flag 
    data_ground: ground truth as an image
    data_pred: ML model prediction as an image
    qlty_flg_temp_slc: quality flag for a single day  """
    h, w = data_ground.shape[0], data_ground.shape[1]
    diff_str = 0    #accumulate mae results
    diff_str_mse = 0    #accumulate mse results
    cnt = 0 # initialize counter for valid pixels
    for j in range(0, w):   #read x-axis
        for i in range (0, h):  #read y-axis
            flg = qlty_flg_temp_slc[j, i]  #read quality flag
            if flg == 0: 
                p = data_ground[j, i]  #get ground-truth pixel value
                q = data_pred[j, i]  #get predicted pixel value
                diff_abs = np.abs(p-q)
                diff_str = diff_str + diff_abs  #update results

                diff_sqd = (p-q)**2  #squared difference
                diff_str_mse = diff_str_mse + diff_sqd  #sum of squared difference results

                cnt += 1
            else:
                pass
    if cnt > 0:
        mae = diff_str/cnt
        rmse = np.sqrt(diff_str_mse/cnt)   #compute rmse
    else:
        mae, rmse = np.nan, np.nan
    return mae, rmse




def mae_dataset(full_inp, test_res_full, qlty_flag, lk_bck=None):
    """ function: computes the mean absolute error and absolute pixelwise difference over all samples/days in the entire datasets
    returns: (i) mean absolute over dataset/days, (ii) a stack similar to the original map with the absolute pixelwise difference
    over the dataset/days
    full_inp: original smap 
    test_res_full: gap-filled/predicted smap
    qlty_flag: quality flag data
    lk_bck: look back time steps """

    cnt2 = 0 # initialize counter for all days in the dataset
    mae_str = 0 #store mae for all days in the dataset
    rmse_str = 0 #store rmse for all days in the dataset
    y, x = test_res_full.shape[1], test_res_full.shape[2]

    full_inp_diff_dum = np.full_like(full_inp, fill_value=np.nan)  
    full_inp_diff_dum_zeros = np.full_like(full_inp, fill_value=0)   
     
    diff_str_all = 0    #accumulate mae for all images
    diff_str_mse_all = 0  #accumulate mse for all images
    cnt1_all = 0           #accumulate the count for valid pixels based on the quality flag
    for k in range(test_res_full.shape[0]):   #starting from the look back time step
        data_ground = full_inp[k + lk_bck, :, :]  
        data_pred = test_res_full[k, :, :, :]  
        data_pred = np.reshape(data_pred, newshape=(100, 100))
        diff_str = 0        #accumulate mae results per image
        diff_str_mse = 0    #accumulate mse results
        cnt1 = 0 # initialize counter for valid pixels per day
        for j in range(0, y):   #read y-axis
            for i in range (0, x):  #read x-axis
                flg = qlty_flag[k + lk_bck, :, :][j, i]  #read quality flag
                if flg == 0: 
                    p = data_ground[j, i]  #get ground-truth pixel value
                    q = data_pred[j, i]  #get predicted pixel value
                    diff_abs = np.abs(p-q)  #compute absolute difference
                    diff_str = diff_str + diff_abs  #update sum of absolute difference results

                    diff_sqd = (p-q)**2  #squared difference
                    diff_str_mse = diff_str_mse + diff_sqd  #sum of squared difference results


                    full_inp_diff_dum[k, :, :][j, i] = diff_abs     #set value corresponding absolute difference
                    # full_inp_diff_dum_zeros[k, :, :][j, i] = diff_abs     #set value corresponding absolute difference

                    cnt1 += 1
                else:
                    pass
        diff_str_all += diff_str
        diff_str_mse_all += diff_str_mse
        cnt1_all += cnt1 

    if cnt1_all !=0:    #check that some valid pixels were found
        mae_all = diff_str_all/cnt1_all     #compute mae
        rmse_all = np.sqrt(diff_str_mse_all/cnt1_all)   #compute rmse

    # full_inp_diff = full_inp_diff_dum[lk_bck: len(full_inp_diff_dum), :, :]     #slice out the value entries using the lookback period
    max_abs_error_diff = np.max(full_inp_diff_dum)
    mae_all = np.round(mae_all, decimals=4)
    rmse_all = np.round(rmse_all, decimals=4)
    max_abs_error_diff = np.round(max_abs_error_diff, decimals=4)
    return mae_all, rmse_all, max_abs_error_diff, full_inp_diff_dum




# def mae_dataset(full_inp, test_res_full, qlty_flag, lk_bck=None):
#     """ function: computes the mean absolute error and absolute pixelwise difference over all samples/days in the entire datasets
#     returns: (i) mean absolute over dataset/days, (ii) a stack similar to the original map with the absolute pixelwise difference
#     over the dataset/days
#     full_inp: original smap 
#     test_res_full: gap-filled/predicted smap
#     qlty_flag: quality flag data
#     lk_bck: look back time steps """

#     cnt2 = 0 # initialize counter for all days in the dataset
#     mae_str = 0 #store mae for all days in the dataset
#     rmse_str = 0 #store rmse for all days in the dataset
#     y, x = test_res_full.shape[1], test_res_full.shape[2]

#     full_inp_diff_dum = np.full_like(full_inp, fill_value=np.nan)  
#     full_inp_diff_dum_zeros = np.full_like(full_inp, fill_value=0)     

#     for k in range(test_res_full.shape[0]):   #starting from the look back time step
#         data_ground = full_inp[k + lk_bck, :, :]  
#         data_pred = test_res_full[k, :, :, :]  
#         data_pred = np.reshape(data_pred, newshape=(100, 100))
#         diff_str = 0        #accumulate mae results
#         diff_str_mse = 0    #accumulate mse results
#         cnt1 = 0 # initialize counter for valid pixels per day
#         for j in range(0, y):   #read y-axis
#             for i in range (0, x):  #read x-axis
#                 flg = qlty_flag[k + lk_bck, :, :][j, i]  #read quality flag
#                 if flg == 0: 
#                     p = data_ground[j, i]  #get ground-truth pixel value
#                     q = data_pred[j, i]  #get predicted pixel value
#                     diff_abs = np.abs(p-q)  #compute absolute difference
#                     diff_str = diff_str + diff_abs  #update sum of absolute difference results

#                     diff_sqd = (p-q)**2  #squared difference
#                     diff_str_mse = diff_str_mse + diff_sqd  #sum of squared difference results


#                     full_inp_diff_dum[k, :, :][j, i] = diff_abs     #set value corresponding absolute difference
#                     full_inp_diff_dum_zeros[k, :, :][j, i] = diff_abs     #set value corresponding absolute difference

#                     cnt1 += 1
#                 else:
#                     pass
#         if cnt1 !=0:    #check that some valid pixels were found
#             mae = diff_str/cnt1     #compute mae
#             rmse = np.sqrt(diff_str_mse/cnt1)   #compute rmse

#         mae_str += mae       ##update mae across all days
#         rmse_str += rmse     #update rmse across all days
#         if cnt1 !=0:    #check that for current day, some valid pixels were found
#             cnt2 +=1
#     mae_all = mae_str/cnt2
#     rmse_all = rmse_str/cnt2 
#     # full_inp_diff = full_inp_diff_dum[lk_bck: len(full_inp_diff_dum), :, :]     #slice out the value entries using the lookback period
#     max_abs_error_diff = np.max(full_inp_diff_dum_zeros)
#     return mae_all, rmse_all, max_abs_error_diff, full_inp_diff_dum





def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def get_valid_days(X_test_full, indx=None):
    """ function: scans a single temporalized sample, and returns only days with at least a valid pixel value using quality flags for checks 
    X-test_full: temporalized dataset
    indx: index of the sample to process """
    ts_data = X_test_full[indx, :, :, :, :]
    all_days = np.empty([X_test_full.shape[2], X_test_full.shape[3], 1])   #creates an empty
    valid_days = all_days[None, :, :, :]   #adds an extra dimension; for saving external loop results
    for i in range(ts_data.shape[0]):
        s = ts_data[i, :, :, :]  #iterate over time/days
        if np.max(s) < 0:
            pass
        else:
            valid_days = np.concatenate((valid_days, s[None, :, :, :]), axis=0)   #save cropped pixels/sm

    valid_days = valid_days[1:, :, :, :]   #slice out valid data from concatenated result
    valid_days = np.expand_dims(valid_days, axis=0)  # Add the batch dimension since we have just one sample
    return valid_days


def get_valid_days_full(full_inp, LOOK_BACK=None, indx=None):
    """ function: scans a single temporalized sample, and returns only days with at least a valid pixel value using quality flags for checks 
    X-test_full: temporalized dataset
    indx: index of the sample to process """
    # ts_data = full_inp[indx+ LOOK_BACK, :, :]
    all_days = np.empty([full_inp.shape[1], full_inp.shape[2]])   #creates an empty
    valid_days = all_days[None, :, :]   #adds an extra dimension; for saving external loop results
    i = 0
    cnt = 0
    exit = False
    while i < full_inp.shape[0] and not exit:
        s = full_inp[indx + i, :, :]  #iterate over time/days
        if np.max(s) < 0:
            pass
        else:
            valid_days = np.concatenate((valid_days, s[None, :, :]), axis=0)   #save cropped pixels/sm
            cnt += 1
        if cnt == LOOK_BACK+1: #check number of collected time-steps is equal to the look back, including the current day
            exit = True
        i += 1

    valid_days = valid_days[1:, :, :]   #slice out valid data from concatenated result
    valid_days = temporalize_test_now(valid_days, lookback=LOOK_BACK) #create/temporalize corresponding quality flags
    valid_days = np.expand_dims(valid_days, axis=-1)  # Add a channel dimension since the images are grayscale
    return valid_days


def valid_sm_single_day(data, qlty_flg_temp_slc):
    """ function: returns valid pixels/sm values using supplied data and quality flag as reference
    data: data to check for valid pixels/sm values
    qlty_flg_temp_slc: quality flag for a single day that is used for validity check  """
    h, w = data.shape[0], data.shape[1]
    diff_str = 0    #accumulate results
    cnt = 0 # initialize counter for valid pixels
    dt_lst = []     #store results
    for j in range(0, w):   #read x-axis
        for i in range (0, h):  #read y-axis
            flg = qlty_flg_temp_slc[j, i]  #read quality flag
            if flg == 0: 
                p = data[j, i]  #get ground-truth pixel value
                dt_lst += [p]   #concatenate results
                cnt += 1
            else:
                pass
    return np.array(dt_lst), cnt


def pix_grd_vld_time(x_grnd_slc, qlty_flg_slc):
    """ function: a vector of ground-truth sm, and returns only the valid sm using the quality flag provided
    x_grnd_slc: vector of ground-truth sm for a specific pixel
    qlty_flg_slc: vector of corresponding quality flag for the ground-truth vector """
    pix_grnd_vld = []
    for i in range(qlty_flg_slc.shape[0]):  #scans along time axis
        if qlty_flg_slc[i] == 0:
            pix_grnd_vld += [x_grnd_slc[i]]    #concatenate value to list
        else:
            pix_grnd_vld += [np.nan]    #concatenate NAN to list for gaps during plotting
    return np.array(pix_grnd_vld)





def singleDay_modelTestwithTime(loaded_model, X_test_full, X_test_time, qlty_flg_temp, proc_day=None, lk_fwd = None):
    """ function: test trained model using the index of the given day. Returns the ground-truth for the speicified day,
    ML model gap-filled result and corresponding quality flag   
    loaded_model: trained model
    X_test_full: test data; single data sample or batched data
    qlty_flg_temp: quality flag for test data
    proc_day: index of day to process; note that this is the day just after the look-back period """ 
    
    lk_fwd_upd = lk_fwd + 1     #update so that retrival is correct
    if X_test_full.ndim == 4:  #check that a single sample is supplied
        sm_sample_days = X_test_full
        sm_sample_days = np.expand_dims(sm_sample_days, axis=0)  # Add the batch dimension since we have just one sample
        tm_sample_days = X_test_time
        tm_sample_days = np.expand_dims(tm_sample_days, axis=0)  # Add the batch dimension since we have just one sample

        #########################
        data_ground = X_test_full[-lk_fwd_upd, :, :, :]   #get a specific current day
        data_ground = np.reshape(data_ground, newshape=(data_ground.shape[0], data_ground.shape[1]))    #reshape to the resolution of sm
        qlty_flg_temp_slc = qlty_flg_temp[-lk_fwd_upd, :, :, :]  #get corresponding quality flag for a specific day. Note that channel was not created for the quality flag, as it is noted used for model training
        qlty_flg_temp_slc = np.reshape(qlty_flg_temp_slc, newshape=(qlty_flg_temp_slc.shape[0], qlty_flg_temp_slc.shape[1])) #reshape to the resolution of sm
        h, w = X_test_full.shape[1], X_test_full.shape[2]

    elif X_test_full.ndim == 5 :  #check that batch data is supplied                         #
        sm_sample_days = X_test_full[proc_day, :, :, :, :]     #get all days which serve as input
        sm_sample_days = np.expand_dims(sm_sample_days, axis=0)  # Add the batch dimension since we have just one sample

        tm_sample_days = X_test_time[proc_day, :, :, :, :]     #get all days which serve as input
        tm_sample_days = np.expand_dims(tm_sample_days, axis=0)  # Add the batch dimension since we have just one sample

        #########################
        data_ground = X_test_full[proc_day, -lk_fwd_upd, :, :, :]   #get a specific current day
        data_ground = np.reshape(data_ground, newshape=(data_ground.shape[0], data_ground.shape[1]))    #reshape to the resolution of sm
        qlty_flg_temp_slc = qlty_flg_temp[proc_day, -lk_fwd_upd, :, :]  #get corresponding quality flag for a specific day. Note that channel was not created for the quality flag, as it is noted used for model training
        qlty_flg_temp_slc = np.reshape(qlty_flg_temp_slc, newshape=(qlty_flg_temp_slc.shape[0], qlty_flg_temp_slc.shape[1])) #reshape to the resolution of sm
        h, w = X_test_full.shape[2], X_test_full.shape[3]
    else:
        raise RuntimeError('expects the number of dimension for testing data to be 4 or 5')
    
    test_res = loaded_model.predict([sm_sample_days, tm_sample_days] , batch_size=1)   #test trained model
    test_res = np.reshape(test_res, newshape=(h, w))

    ############################using the valid days check
    # valid_days = get_valid_days(X_test_full, indx= proc_day)    #valid days is returned with batch dimension added
    # valid_days = get_valid_days_full(full_inp, LOOK_BACK=7, indx=proc_day)
    # test_res_checks = loaded_model.predict(valid_days, batch_size=1)   #test trained model
    # test_res_checks = np.reshape(test_res_checks, newshape=(100, 100))

    #########################
    # data_ground = X_test_full[proc_day, -1, :, :, :]   #get a specific current day
    # data_ground = np.reshape(data_ground, newshape=(data_ground.shape[0], data_ground.shape[1]))    #reshape to the resolution of sm
    # qlty_flg_temp_slc = qlty_flg_temp[proc_day, -1, :, :]  #get corresponding quality flag for a specific day. Note that channel was not created for the quality flag, as it is noted used for model training
    # qlty_flg_temp_slc = np.reshape(qlty_flg_temp_slc, newshape=(qlty_flg_temp_slc.shape[0], qlty_flg_temp_slc.shape[1])) #reshape to the resolution of sm
    
    return data_ground, test_res, qlty_flg_temp_slc



def singleDay_modelTest(loaded_model, X_test_full, qlty_flg_temp, proc_day=None):
    """ function: test trained model using the index of the given day. Returns the ground-truth for the speicified day,
    ML model gap-filled result and corresponding quality flag   
    loaded_model: trained model
    X_test_full: test data; single data sample or batched data
    qlty_flg_temp: quality flag for test data
    proc_day: index of day to process; note that this is the day just after the look-back period """ 
    
    if X_test_full.shape[0] == 1:  #check that a single sample is supplied
        sample_days = X_test_full
    elif len(X_test_full.shape) > 1 :  #heck that batch data is supplied                         #
        sample_days = X_test_full[proc_day, :, :, :, :]     #get all days which serve as input
        sample_days = np.expand_dims(sample_days, axis=0)  # Add the batch dimension since we have just one sample
    
    test_res = loaded_model.predict(sample_days, batch_size=1)   #test trained model
    test_res = np.reshape(test_res, newshape=(100, 100))

    ############################using the valid days check
    # valid_days = get_valid_days(X_test_full, indx= proc_day)    #valid days is returned with batch dimension added
    # valid_days = get_valid_days_full(full_inp, LOOK_BACK=7, indx=proc_day)
    # test_res_checks = loaded_model.predict(valid_days, batch_size=1)   #test trained model
    # test_res_checks = np.reshape(test_res_checks, newshape=(100, 100))

    #########################
    data_ground = X_test_full[proc_day, -1, :, :, :]   #get a specific current day
    data_ground = np.reshape(data_ground, newshape=(data_ground.shape[0], data_ground.shape[1]))    #reshape to the resolution of sm
    qlty_flg_temp_slc = qlty_flg_temp[proc_day, -1, :, :]  #get corresponding quality flag for a specific day. Note that channel was not created for the quality flag, as it is noted used for model training
    qlty_flg_temp_slc = np.reshape(qlty_flg_temp_slc, newshape=(qlty_flg_temp_slc.shape[0], qlty_flg_temp_slc.shape[1])) #reshape to the resolution of sm
    
    return data_ground, test_res, qlty_flg_temp_slc


def pixels_timeSeries(X_test_full, test_res_full, qlty_flg_temp, pix_idx=None):
    """ function: returns the flattend time series for both ground-truth and gap-filled using specified pixel index 
    X_test_full: complete test input data
    test_res_full: complete gap-filled dataset
    qlty_flg_temp: complete quality flag data
    pix_idx: index of pixel to process  """
    pix_res_slc = np.reshape(test_res_full, 
    newshape=(test_res_full.shape[0], test_res_full.shape[1] * test_res_full.shape[2])) #flatten daily sm/pixels 
    pix_res_slc = pix_res_slc[:, pix_idx]  #get all daily predicted sm for a specific pixel i

    # x_grnd_slc  = X_test_full
    x_grnd_slc = np.reshape(X_test_full, 
    newshape=(X_test_full.shape[0], X_test_full.shape[1] * X_test_full.shape[2])) #flatten daily sm/pixels
    x_grnd_slc = x_grnd_slc[:, pix_idx]  #get all daily ground-truth sm for a specific pixel i

    # qlty_flg_slc = qlty_flg_temp[:, -1, :, :]
    qlty_flg_slc = np.reshape(qlty_flg_temp, 
    newshape=(qlty_flg_temp.shape[0], qlty_flg_temp.shape[1] * qlty_flg_temp.shape[2])) #flatten daily quality flags
    qlty_flg_slc = qlty_flg_slc[:, pix_idx]  #get all daily predicted sm for a specific pixel i

    pix_grnd_vld = pix_grd_vld_time(x_grnd_slc, qlty_flg_slc)   #get valid sm for a single pixel over the time-series length

    return pix_grnd_vld, pix_res_slc

def add_gap_day(X_test_full, y, x, y_end, x_end, proc_day=None, fill_val=None):
    """ function: add artificial gaps to data 
    y: starting row to start adding artificial gap 
    x: starting column to start adding artificial gap 
    y_end: starting row to end adding artificial gap 
    x_end: starting column to end adding artificial gap 
    proc_day: index of the day in the dataset to process
    fill_value: value used for filling the added artificial gap  """
    y_mk_sz = y_end - y
    x_mk_sz = x_end - x
    mask_sim = np.ones(shape=(y_mk_sz, x_mk_sz)) * fill_val #simulate/generate a mask filled with all values of -0.01 
    X_test_full_copy = copy.deepcopy(X_test_full) 
    X_test_full_copy[proc_day, -1, :, :, 0][y: y + y_end, x: x + x_end] = mask_sim  #over-write specified portion with mask
    return X_test_full_copy 


def post_proc_smap_res_dataset(test_res_full, qlty_flag, lk_bck):
    """ function: sets all non-valid pixels over the entire gap-filled dataset results to 
    nan using the supplied qlty flag
    test_res_full: array, which is the entire dataset gap-filled
    qlty_flag: array, which is the entire dataset quality flag for all pixels 
    lk_bck: integer, look back time step  """

    test_res_full_post = copy.deepcopy(test_res_full)   #performs deep copy so that orignal data is no longer referenced
    t = test_res_full_post.shape[0]
    h = test_res_full_post.shape[1]
    w = test_res_full_post.shape[2]
    test_res_full_post = np.reshape(test_res_full_post, newshape=(t, h, w))  #remove the channel axis introduce during model inference
    y, x = test_res_full_post.shape[1], test_res_full_post.shape[2]

    for k in range(test_res_full_post.shape[0]):   #loop through data
        # cnt = 0 # initialize counter for valid pixels per day
        for j in range(0, y):   #read y-axis
            for i in range (0, x):  #read x-axis
                flg = qlty_flag[k + lk_bck, :, :][j, i]  #read quality flag starting from the look back time step
                if flg != 0: 
                    test_res_full_post[k, :, :][j, i] = np.nan    #set pixel to nan
                else:
                    # cnt += 1
                    pass
                    
    test_res_full_post = np.expand_dims(test_res_full_post, axis=-1)  # add back a channel dimension
    return test_res_full_post


def post_proc_smap_res(test_res, ground_res, qlty_flag_slc):
    """ function: sets all non-valid pixels in a single gap-filled map/image results to 
    nan using the supplied qlty flag
    test_res_full: array, which is the entire dataset gap-filled
    qlty_flag: array, which is the entire dataset quality flag for all pixels 
    lk_bck: integer, look back time step  """
    ground_res_post = ground_res.copy()   #performs deep copy so that orignal data is no longer referenced
    test_res_post = test_res.copy()    #performs deep copy so that orignal data is no longer referenced
    y, x = test_res_post.shape[0], test_res_post.shape[1]  
    for j in range(0, y):   #read y-axis
        for i in range (0, x):  #read x-axis
            flg = qlty_flag_slc[j, i]  #read quality flag starting from the look back time step
            if flg != 0: 
                ground_res_post[j, i] = np.nan    #set pixel to nan
                test_res_post[j, i] = np.nan    #set pixel to nan
            else:
                pass
                    
    return ground_res_post, test_res_post



def data_btwTime_idx(smap_data, sm_time, start_idx=None, lk_bck=7, 
                        lbk_time = 9, lk_fwd = 3, lk_fwd_time = 5, lbk_days_fmt = 'hrs'):
    """ function: uses the supplied index to start tracking which data in the smap 
    data that can be captured in the given look back time supplied 
    smap_data: full smap data array 
    sm_time: full acquisition time array for smap data
    start_idx: starting index that determines the start of looking back 
    max_lk_bck: max number of measurements/images to look back to collect and return irrespective of the number of images observed
    lbk_time: the actual number of days that must have elapsed before look back ends 
    lbk_days_fmt: time format """ 
        
    # upd_start_idx = start_idx + lk_bck  #can only start from the look back index
    # start_time = sm_time[upd_start_idx]    #retrieve the start time using index
    # end_time = start_time - look_back_time

    # if max_lk_bck < lbk_time:
    #     print('you supplied function argument "lk_bck" that is less than "lbk_days" ')
    #     raise RuntimeError('the number of look back measurements should be greater than the actual number of look back days')
    
    if lbk_days_fmt== 'secs':
        look_back_time = np.timedelta64(lbk_time, 's')
        look_forward_time = np.timedelta64(lk_fwd_time, 's')
    elif lbk_days_fmt== 'mins':
        look_back_time = np.timedelta64(lbk_time, 'm')
        look_forward_time = np.timedelta64(lk_fwd_time, 'm')
    elif lbk_days_fmt== 'hrs':
        look_back_time = np.timedelta64(lbk_time, 'h')
        look_forward_time = np.timedelta64(lk_fwd_time, 'h')
    elif lbk_days_fmt== 'days':
        look_back_time = np.timedelta64(lbk_time, 'D')
        look_forward_time = np.timedelta64(lk_fwd_time, 'D')
    else:
        raise RuntimeError('the supplied time format is unknown')

    lk_bck_upd = lk_bck + 1 #update so that the measurements/images starting current measurement aligns with look back specified

    # start_idx_upd = start_idx + lk_bck
    start_idx_upd = start_idx
    current_time = sm_time[start_idx_upd]    #current time is the acquisition time for the current image to be gap-filled
    end_time = current_time - look_back_time
    fut_time = current_time + look_forward_time     #maximum time in the future to extract data


    # end_time = sm_time[start_idx]
    # current_time = end_time + look_back_time    #current time is the acquisition time for the current image to be gap-filled

    re_smap_dum = np.empty([smap_data.shape[1], 
                                smap_data.shape[2]])   #creates an empty
    re_smap_dum = re_smap_dum[None, :, :]   #adds an extra dimension; for saving external loop results

    re_smap_dum_fwd = np.empty([smap_data.shape[1], 
                                smap_data.shape[2]])   #creates an empty
    re_smap_dum_fwd = re_smap_dum_fwd[None, :, :]   #adds an extra dimension; for saving external loop results

    idx_cnt = 0             #index counter for tracking slices to extract         
    
    tm_per_inval = np.datetime64('2016-12-30T00:00:00.000000000')        #invalid time for invalid measurements
    if start_idx_upd + lk_fwd < len(sm_time):
        # if diff_chk <= look_back_time:                  #check that measurements taken do not fall outside of range
            # for i in range(lk_bck +1): #start looking back from the start index
        for i in range(start_idx_upd + 1): #start looking back from the start index
            if sm_time[start_idx_upd - i] >= end_time:
                # print('index:', start_idx_upd - i)
                smap_slc = smap_data[start_idx_upd - i, :, :]   #extract corresponding array in smap data
                re_smap_dum = np.concatenate((smap_slc[None, :, :], re_smap_dum), axis=0)   #append from the left
            
                idx_cnt += 1                    #update index counter
            
            elif sm_time[start_idx_upd - i] < tm_per_inval:     #pass days which are less than reasonable time
                pass
            else:
                break
                # pass
        
        if len(re_smap_dum) != 0:
            re_smap_full = re_smap_dum[:-1, :, :]     #slice out array excluding the last dummy at index -1
            if len(re_smap_full) > lk_bck_upd:
                ln_cut = len(re_smap_full) - lk_bck_upd
                re_smap_full = re_smap_full[ln_cut:, :, :]  #extract only the look back period
            for j in range(start_idx_upd +1 , len(smap_data)):  #start from the measurement/image following the current one
                if sm_time[j] <= fut_time:
                    smap_slc = smap_data[j, :, :]   #extract corresponding array in smap data
                    re_smap_dum_fwd = np.concatenate((re_smap_dum_fwd, smap_slc[None, :, :]), axis=0)   #append to collected measurements from the right
                
                elif sm_time[start_idx_upd - i] < tm_per_inval:     #pass days which are less than reasonable time
                    pass
                else:
                    break
                # else:
                #     break
            re_smap_dum_fwd = re_smap_dum_fwd[1:, :, :]     #slice out array excluding the first index  
            if len(re_smap_dum_fwd) > lk_fwd:
                re_smap_dum_fwd = re_smap_dum_fwd[:lk_fwd, :, :]     #slice out array excluding the first index
            re_smap_full = np.concatenate((re_smap_full, re_smap_dum_fwd), axis=0)   #ajoin left and right smaps extracted
            last_tm = sm_time[start_idx_upd]            #time for current measurement/image
        else:
            re_smap_full = []
            last_tm = []
    else:
        re_smap_full = []
        last_tm = []
    return re_smap_full, last_tm




def data_bwtTime_dataset(smap_data, sm_time, lk_bck=None, 
                        lk_bck_time = np.timedelta64(10, 'D')):

    re_smap_dum = np.empty([smap_data.shape[1], 
                                smap_data.shape[2]])   #creates an empty
    re_smap_dum = re_smap_dum[None, :, :]   #adds an extra dimension; for saving external loop results

    for i in range(len(smap_data) - 20):
    # for i in range(len(smap_data) - lk_bck-1):
        print('i:', i)
        smap_slc, _ = data_btwTime_idx(smap_data, sm_time, start_idx=i, lk_bck=lk_bck, 
                    look_back_time = lk_bck_time)

        re_smap_dum = np.concatenate((re_smap_dum, smap_slc), axis=0)
        # re_smap_dum = np.concatenate((re_smap_dum, smap_slc[None, :, :]), axis=0)

    re_smap = re_smap_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
    return re_smap


def apply_static_qlty_flag(data_dum, static_flag):
    """ function: applies pre-computed static quality flags on the input data 
    data_dum: input data to be processed 
    static_flag: pre-computed static flag """ 

    data = data_dum.copy()
    y, x = data.shape[0], data.shape[1]

    for j in range(0, y):   #read y-axis
        for i in range (0, x):  #read x-axis
            flg = static_flag[j, i]  #read quality flag starting from the look back time step
            if flg != 0: 
                data[j, i] = np.nan    #set pixel to nan
            else:
                pass
    return data


def static_qlty_flag_mask(qlty_flg):
    h, w = qlty_flg.shape[0], qlty_flg.shape[1]
    diff_str = 0    #accumulate mae results
    diff_str_mse = 0    #accumulate mse results
    cnt = 0 # initialize counter for valid pixels
    bs_slc = qlty_flg[0, : , :]     #base slice
    for k in range(1, len(qlty_flg)):   #scan along time axis
        slc_upd = bs_slc * qlty_flg[k, : , :]
        bs_slc = slc_upd
        bs_slc = np.clip(bs_slc, a_min=0, a_max=1)   #clip array to binary format
    return bs_slc


def cal_qflag(time_diff_idx, weights, arr_len):
    weights = weights[0: arr_len]
    res = np.sum(np.array(time_diff_idx) * np.array(weights))/ np.sum(weights)
    res = res * 100
    return res


def comp_pix_qlty(sm, tm, lookback=9, max_num_per_day=2):
    """ function: computes per pix quality based on time difference and 
    sm: soil moisture array 
    lookback: number of measurements to look back 
    max_num_per_day: maximum number of measurements per day """ 

    sm_flat = np.reshape(sm, (sm.shape[0], sm.shape[1]*sm.shape[2]))
    tm_flat = np.reshape(tm, (tm.shape[0], tm.shape[1]*tm.shape[2]))
    
    y, x = sm_flat.shape
    pix_qlty_2d = np.full(shape= (1, x), fill_value=0)
    # print('pix_qlty_2d shape:', pix_qlty_2d.shape)

    # lookback = 9  # number of days in the past
    # max_num_per_day = 2
    time_diff_list = np.arange(lookback * max_num_per_day) / float(max_num_per_day)
    weights = np.array([np.exp(-(t / (lookback / 2.0))) for t in time_diff_list])

    for i in range(x):
        sm_flat_slc = sm_flat[:, i]
        tm_flat_slc = tm_flat[:, i]/(np.max(tm_flat[:, i]) + 1e-5)
        tm_flat_slc[sm_flat_slc < 0] = 0    #set to zero where in tm_flat_slc, sm_flat_slc is less than zero.

        tm_len = len(tm_flat_slc)
        pix_qlty = cal_qflag(tm_flat_slc, weights, tm_len)
        pix_qlty_2d[:, i] = pix_qlty
    
    pix_qlty_2d = np.reshape(pix_qlty_2d, (sm.shape[1], sm.shape[2]))

    return pix_qlty_2d/100.0


def var_len_testModel_fut(loaded_model, full_inp, time_3d_fmt, qlty_flag, static_flag, time_1d, LOOK_BACK, LOOK_BACK_TIME_TEST, 
    TIME_FORMAT, LOOK_FORWARD, LOOK_FORWARD_TIME, rot_stack=True, applyStatFlag= False, time_1d_str_form = None, 
    tile_name= None, comp_pixSeriesQlty= True, plot_operations= None, plot_cmap= None, DISP_PLOT= True, SAVE_PLOTS= True, 
    save_plot_path= None):
    """ function: takes smap and acquistion time, tests the ML model, plot gap-filled results. Returns arrays of 
    the predicted outputs and 1d times for every sample tested 
    loaded_model: trained ML model to use for testing
    full_inp: smap 3d data for testing 
    time_3d_fmt: smap times in 3d for testing 
    qlty_flag: quality flag in 3d for testing 
    time_1d: actual and precise smap acquisition time in 1d array
    LOOK_BACK: number of measurements in the past to include in the testing 
    LOOK_BACK_TIME_TEST: actual time in the past to collect smap data for testing 
    TIME_FORMAT: time format in 'mins', 'hrs' or 'days' 
    DISP_PLOT: flag for showing plots 
    SAVE_PLOTS: flag for saving plots """

    gap_filled = np.empty([full_inp.shape[1], 
                                    full_inp.shape[2]])   #creates an empty
    gap_filled  = gap_filled[None, :, :]   #adds an extra dimension; for saving external loop results

    ground_dum = np.empty([full_inp.shape[1], 
                                    full_inp.shape[2]])   #creates an empty
    ground_dum = ground_dum[None, :, :]   #adds an extra dimension; for saving external loop results 

    qltyFlag_dum = np.empty([full_inp.shape[1], 
                                    full_inp.shape[2]])   #creates an empty
    qltyFlag_dum = qltyFlag_dum[None, :, :]   #adds an extra dimension; for saving external loop results     

    pix_qly_2d_dum = np.empty([100, 
                                    100])   #creates an empty
    pix_qly_2d_dum = pix_qly_2d_dum[None, :, :]   #adds an extra dimension; for saving external loop results                        
                        

    time_keep_1d = []

    if rot_stack==True:
        static_flag = rotate_stack(static_flag)
    
    LOOK_FORWARD_UPD = LOOK_FORWARD + 1     #update so that retrival is correct
    static_flag_cut = static_flag[25: 125, 25: 125]

    # for i in range(0, 25):
    for i in range(len(full_inp)):
        print('testing sample i:', i)
        smap_var_len, time_curr_1d = data_btwTime_idx(full_inp, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
                            lbk_time = LOOK_BACK_TIME_TEST, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, 
                            lbk_days_fmt = TIME_FORMAT)

        qlty_flag_var_len, _ = data_btwTime_idx(qlty_flag, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
                            lbk_time = LOOK_BACK_TIME_TEST, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, 
                            lbk_days_fmt = TIME_FORMAT)

        time_var_len_3d, _ = data_btwTime_idx(time_3d_fmt, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
                            lbk_time = LOOK_BACK_TIME_TEST, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, 
                            lbk_days_fmt = TIME_FORMAT)
        
        measur_len = len(smap_var_len)          #get number of measurements/images retrieved; max will be the look back time steps
        
        # print('test_res shape:', smap_var_len.shape)
        # print('qlty_flag_var_len shape:', qlty_flag_var_len.shape)
        # print('time_var_len_3d:', time_var_len_3d.shape)

        if measur_len == 0:  #if an empty list is returned; data sequence unsuccessful
            smap_var_len = full_inp[i]
            time_var_len_3d = time_3d_fmt[i]
            qlty_flag_var_len = qlty_flag[i]

        ###########compute time difference
        time_diff_var_len = time_diff(time_var_len_3d, lk_fwd = LOOK_FORWARD)

        ######normalize time data
        time_diff_var_len = time_diff_var_len/LOOK_BACK_TIME_TEST  
        # time_diff_var_len = time_diff_var_len/11                   

        smap_var_len_ch = smap_var_len[:, 25: 125, 25: 125] #take 25:125 when smap input is cut 75:175
        time_diff_var_len_ch = time_diff_var_len[:, 25: 125, 25: 125] #take 25:125 when smap input is cut 75:175
        # smap_var_len_ch = smap_var_len[:, 50: 150, 50: 150] #take 50:150 when smap input is cut 50:250
        # time_diff_var_len_ch = time_diff_var_len[:, 50: 150, 50: 150] #take 50:150 when smap input is cut 50:250
        pix_qly_2d = comp_pix_qlty(smap_var_len_ch, time_diff_var_len_ch, lookback=9, max_num_per_day=2)                     

        ######add channels to inputs
        X_test_full = np.expand_dims(smap_var_len, axis=-1) 
        X_test_time = np.expand_dims(time_diff_var_len, axis=-1)
        qlty_flg_temp = np.expand_dims(qlty_flag_var_len, axis=-1) 

        # print('X_test_full shape:', X_test_full.shape)

        ########test trained ml model
        if measur_len < (LOOK_BACK + LOOK_FORWARD):     #check that there's suitable number of measurements for testing
            if measur_len < 2:               #check that the current time step measurement can be retrieved
                LOOK_FORWARD_UPD = 1
            data_ground = X_test_full[-LOOK_FORWARD_UPD, :, :, :]   #get a specific current day
            data_ground = np.reshape(data_ground, newshape=(data_ground.shape[0], data_ground.shape[1]))    #reshape to the resolution of sm

            test_res = data_ground     #set as ground-truth, since we do no gap-filling

            qlty_flg_temp_slc = qlty_flg_temp[-LOOK_FORWARD_UPD, :, :, :]
            qlty_flg_temp_slc = np.reshape(qlty_flg_temp_slc, newshape=(qlty_flg_temp_slc.shape[0], qlty_flg_temp_slc.shape[1]))

        else:
            data_ground, test_res, qlty_flg_temp_slc = singleDay_modelTestwithTime(loaded_model, 
            X_test_full, X_test_time, qlty_flg_temp, proc_day= i, lk_fwd = LOOK_FORWARD)
        
        if rot_stack== True:
            data_ground = rotate_stack(data_ground, rot_axis= (0, 1))
            test_res = rotate_stack(test_res, rot_axis= (0, 1))
            qlty_flg_temp_slc = rotate_stack(qlty_flg_temp_slc, rot_axis= (0, 1))


        data_ground_cut = data_ground[25: 125, 25: 125]  #take 25:125 when smap input is cut 75:175
        test_res_cut = test_res[25: 125, 25: 125]   #take 25:125 when smap input is cut 75:175
        qlty_flg_temp_slc_cut = qlty_flg_temp_slc[25: 125, 25: 125]   #take 25:125 when smap input is cut 75:175

        
        
        ####stack gap-filled smaps and ground-truth smaps
        gap_filled = np.concatenate((gap_filled, test_res[None, :, :]), axis=0)
        # ground_dum = np.concatenate((ground_dum, data_ground[None, :, :]), axis=0)
        # qltyFlag_dum = np.concatenate((qltyFlag_dum, qlty_flg_temp_slc[None, :, :]), axis=0)

        if comp_pixSeriesQlty == True: 
            pix_qly_2d_dum = np.concatenate((pix_qly_2d_dum, pix_qly_2d[None, :, :]), axis=0)

        ####stack time
        time_keep_1d.append(time_curr_1d)
        mae_day, rmse_day = mae_rmse_single_day(data_ground_cut, test_res_cut, qlty_flg_temp_slc_cut) #mae on a specific day without checking valid days
        if mae_day != np.nan:       #check for days in which the quality flag indicates no valid measurment
            mae_day = np.round(mae_day, decimals=4)
            data_ground_cut_post, test_res_cut_post = post_proc_smap_res(test_res_cut, data_ground_cut, 
                                                qlty_flg_temp_slc_cut)   #apply qlty flags on gap-filled results

            if applyStatFlag== True:
                test_res_cut = apply_static_qlty_flag(test_res_cut, static_flag_cut)
            
            if plot_operations==True:
                if time_1d_str_form !=None:
                    fig_name = tile_name + '_' + time_1d_str_form[i]        #put together name for saving plotted figure
                    plotMap_same_scale(data_ground_cut, test_res_cut, plot_title='Results: ' + 
                            time_1d_str_form[i] + ' with ' + str(measur_len)+ ' measurements. ' + 'MAE is '+ str(mae_day), sv_plt=SAVE_PLOTS, 
                            sv_path=save_plot_path, sv_idx= fig_name, custom_cmap=plot_cmap, 
                            show_plot= DISP_PLOT, lk_bck_days= LOOK_BACK_TIME_TEST)  #note that the preprocessed gap-filled result is plotted
                else:
                    plotMap_same_scale(data_ground_cut_post, test_res_cut, plot_title='Results: ' + 
                        str(i) + ' with ' + str(measur_len)+ ' measurements. ' + 'MAE: '+ str(mae_day), sv_plt=SAVE_PLOTS, 
                        sv_path=save_plot_path, sv_idx=i, custom_cmap=plot_cmap, 
                        show_plot= DISP_PLOT, lk_bck_days= LOOK_BACK_TIME_TEST)  #note that the preprocessed gap-filled result is plotted

    gap_filled = gap_filled[1:, :, :]     #slice out array excluding the first dummy at index 0
    # ground_dum = ground_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
    # qltyFlag_dum = qltyFlag_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
    # print('test_res shape:', gap_filled.shape)
    # print('data_ground shape:', ground_dum.shape)
    # print('qlty_flg_temp_slc shape:', qltyFlag_dum.shape)
    if comp_pixSeriesQlty == True: 
        pix_qly_2d_dum = pix_qly_2d_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
    else:
        pix_qly_2d_dum = []

    time_keep_1d = np.array(time_keep_1d)
    return gap_filled, pix_qly_2d_dum





# def var_len_testModel_fut(loaded_model, full_inp, time_3d_fmt, qlty_flag, static_flag, time_1d, LOOK_BACK, LOOK_BACK_TIME_TEST, 
#     TIME_FORMAT, LOOK_FORWARD, LOOK_FORWARD_TIME, rot_stack=True, applyStatFlag= False, time_1d_str_form = None, 
#     tile_name= None, comp_pixSeriesQlty= True, plot_operations= None, plot_cmap= None, DISP_PLOT= True, SAVE_PLOTS= True, 
#     save_plot_path= None):
#     """ function: takes smap and acquistion time, tests the ML model, plot gap-filled results. Returns arrays of 
#     the predicted outputs and 1d times for every sample tested 
#     loaded_model: trained ML model to use for testing
#     full_inp: smap 3d data for testing 
#     time_3d_fmt: smap times in 3d for testing 
#     qlty_flag: quality flag in 3d for testing 
#     time_1d: actual and precise smap acquisition time in 1d array
#     LOOK_BACK: number of measurements in the past to include in the testing 
#     LOOK_BACK_TIME_TEST: actual time in the past to collect smap data for testing 
#     TIME_FORMAT: time format in 'mins', 'hrs' or 'days' 
#     DISP_PLOT: flag for showing plots 
#     SAVE_PLOTS: flag for saving plots """

#     gap_filled = np.empty([full_inp.shape[1], 
#                                     full_inp.shape[2]])   #creates an empty
#     gap_filled  = gap_filled[None, :, :]   #adds an extra dimension; for saving external loop results

#     ground_dum = np.empty([full_inp.shape[1], 
#                                     full_inp.shape[2]])   #creates an empty
#     ground_dum = ground_dum[None, :, :]   #adds an extra dimension; for saving external loop results 

#     qltyFlag_dum = np.empty([full_inp.shape[1], 
#                                     full_inp.shape[2]])   #creates an empty
#     qltyFlag_dum = qltyFlag_dum[None, :, :]   #adds an extra dimension; for saving external loop results     

#     pix_qly_2d_dum = np.empty([100, 
#                                     100])   #creates an empty
#     pix_qly_2d_dum = pix_qly_2d_dum[None, :, :]   #adds an extra dimension; for saving external loop results                        
                        

#     time_keep_1d = []

#     if rot_stack==True:
#         static_flag = rotate_stack(static_flag)
    
#     LOOK_FORWARD_UPD = LOOK_FORWARD + 1     #update so that retrival is correct
#     static_flag_cut = static_flag[25: 125, 25: 125]

#     # for i in range(0, 25):
#     for i in range(len(full_inp)):
#         print('testing sample i:', i)
#         smap_var_len, time_curr_1d = data_btwTime_idx(full_inp, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
#                             lbk_time = LOOK_BACK_TIME_TEST, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, 
#                             lbk_days_fmt = TIME_FORMAT)

#         qlty_flag_var_len, _ = data_btwTime_idx(qlty_flag, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
#                             lbk_time = LOOK_BACK_TIME_TEST, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, 
#                             lbk_days_fmt = TIME_FORMAT)

#         time_var_len_3d, _ = data_btwTime_idx(time_3d_fmt, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
#                             lbk_time = LOOK_BACK_TIME_TEST, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, 
#                             lbk_days_fmt = TIME_FORMAT)
        
#         measur_len = len(smap_var_len)          #get number of measurements/images retrieved; max will be the look back time steps
        
#         # print('test_res shape:', smap_var_len.shape)
#         # print('qlty_flag_var_len shape:', qlty_flag_var_len.shape)
#         # print('time_var_len_3d:', time_var_len_3d.shape)

#         if measur_len != 0:
#             ###########compute time difference
#             time_diff_var_len = time_diff(time_var_len_3d, lk_fwd = LOOK_FORWARD)

#             ######normalize time data
#             time_diff_var_len = time_diff_var_len/LOOK_BACK_TIME_TEST  
#             # time_diff_var_len = time_diff_var_len/11                   

#             smap_var_len_ch = smap_var_len[:, 25: 125, 25: 125] #take 25:125 when smap input is cut 75:175
#             time_diff_var_len_ch = time_diff_var_len[:, 25: 125, 25: 125] #take 25:125 when smap input is cut 75:175
#             # smap_var_len_ch = smap_var_len[:, 50: 150, 50: 150] #take 50:150 when smap input is cut 50:250
#             # time_diff_var_len_ch = time_diff_var_len[:, 50: 150, 50: 150] #take 50:150 when smap input is cut 50:250
#             pix_qly_2d = comp_pix_qlty(smap_var_len_ch, time_diff_var_len_ch, lookback=9, max_num_per_day=2)                     

#             ######add channels to inputs
#             X_test_full = np.expand_dims(smap_var_len, axis=-1) 
#             X_test_time = np.expand_dims(time_diff_var_len, axis=-1)
#             qlty_flg_temp = np.expand_dims(qlty_flag_var_len, axis=-1) 

#             # print('X_test_full shape:', X_test_full.shape)

#             ########test trained ml model
#             if measur_len < (LOOK_BACK + LOOK_FORWARD):     #check that there's suitable number of measurements for testing
#                 if measur_len < 2:               #check that the current time step measurement can be retrieved
#                     LOOK_FORWARD_UPD = 1
#                 data_ground = X_test_full[-LOOK_FORWARD_UPD, :, :, :]   #get a specific current day
#                 data_ground = np.reshape(data_ground, newshape=(data_ground.shape[0], data_ground.shape[1]))    #reshape to the resolution of sm

#                 test_res = data_ground     #set as ground-truth, since we do no gap-filling

#                 qlty_flg_temp_slc = qlty_flg_temp[-LOOK_FORWARD_UPD, :, :, :]
#                 qlty_flg_temp_slc = np.reshape(qlty_flg_temp_slc, newshape=(qlty_flg_temp_slc.shape[0], qlty_flg_temp_slc.shape[1]))

#             else:
#                 data_ground, test_res, qlty_flg_temp_slc = singleDay_modelTestwithTime(loaded_model, 
#                 X_test_full, X_test_time, qlty_flg_temp, proc_day= i, lk_fwd = LOOK_FORWARD)
            
#             if rot_stack== True:
#                 data_ground = rotate_stack(data_ground, rot_axis= (0, 1))
#                 test_res = rotate_stack(test_res, rot_axis= (0, 1))
#                 qlty_flg_temp_slc = rotate_stack(qlty_flg_temp_slc, rot_axis= (0, 1))


#             data_ground_cut = data_ground[25: 125, 25: 125]  #take 25:125 when smap input is cut 75:175
#             test_res_cut = test_res[25: 125, 25: 125]   #take 25:125 when smap input is cut 75:175
#             qlty_flg_temp_slc_cut = qlty_flg_temp_slc[25: 125, 25: 125]   #take 25:125 when smap input is cut 75:175

            
            
#             ####stack gap-filled smaps and ground-truth smaps
#             gap_filled = np.concatenate((gap_filled, test_res[None, :, :]), axis=0)
#             # ground_dum = np.concatenate((ground_dum, data_ground[None, :, :]), axis=0)
#             # qltyFlag_dum = np.concatenate((qltyFlag_dum, qlty_flg_temp_slc[None, :, :]), axis=0)

#             if comp_pixSeriesQlty == True: 
#                 pix_qly_2d_dum = np.concatenate((pix_qly_2d_dum, pix_qly_2d[None, :, :]), axis=0)

#             ####stack time
#             time_keep_1d.append(time_curr_1d)
#             mae_day, rmse_day = mae_rmse_single_day(data_ground_cut, test_res_cut, qlty_flg_temp_slc_cut) #mae on a specific day without checking valid days
#             if mae_day != np.nan:       #check for days in which the quality flag indicates no valid measurment
#                 mae_day = np.round(mae_day, decimals=4)
#                 data_ground_cut_post, test_res_cut_post = post_proc_smap_res(test_res_cut, data_ground_cut, 
#                                                     qlty_flg_temp_slc_cut)   #apply qlty flags on gap-filled results

#                 if applyStatFlag== True:
#                     test_res_cut = apply_static_qlty_flag(test_res_cut, static_flag_cut)
                
#                 if plot_operations==True:
#                     if time_1d_str_form !=None:
#                         fig_name = tile_name + '_' + time_1d_str_form[i]        #put together name for saving plotted figure
#                         plotMap_same_scale(data_ground_cut, test_res_cut, plot_title='Results: ' + 
#                                 time_1d_str_form[i] + ' with ' + str(measur_len)+ ' measurements. ' + 'MAE is '+ str(mae_day), sv_plt=SAVE_PLOTS, 
#                                 sv_path=save_plot_path, sv_idx= fig_name, custom_cmap=plot_cmap, 
#                                 show_plot= DISP_PLOT, lk_bck_days= LOOK_BACK_TIME_TEST)  #note that the preprocessed gap-filled result is plotted
#                     else:
#                         plotMap_same_scale(data_ground_cut_post, test_res_cut, plot_title='Results: ' + 
#                             str(i) + ' with ' + str(measur_len)+ ' measurements. ' + 'MAE: '+ str(mae_day), sv_plt=SAVE_PLOTS, 
#                             sv_path=save_plot_path, sv_idx=i, custom_cmap=plot_cmap, 
#                             show_plot= DISP_PLOT, lk_bck_days= LOOK_BACK_TIME_TEST)  #note that the preprocessed gap-filled result is plotted

#     gap_filled = gap_filled[1:, :, :]     #slice out array excluding the first dummy at index 0
#     # ground_dum = ground_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
#     # qltyFlag_dum = qltyFlag_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
#     # print('test_res shape:', gap_filled.shape)
#     # print('data_ground shape:', ground_dum.shape)
#     # print('qlty_flg_temp_slc shape:', qltyFlag_dum.shape)
#     if comp_pixSeriesQlty == True: 
#         pix_qly_2d_dum = pix_qly_2d_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
#     else:
#         pix_qly_2d_dum = []

#     time_keep_1d = np.array(time_keep_1d)
#     return gap_filled, pix_qly_2d_dum











# def var_len_testModel(loaded_model, full_inp, time_3d_fmt, qlty_flag, time_1d, LOOK_BACK, LOOK_BACK_TIME, TIME_FORMAT,
#             LOOK_FORWARD, LOOK_FORWARD_TIME, plot_operations= None, plot_cmap= None, 
#             DISP_PLOT= True, SAVE_PLOTS= True, save_plot_path= None):
#     """ function: takes smap and acquistion time, tests the ML model, plot gap-filled results. Returns arrays of 
#     the predicted outputs and 1d times for every sample tested 
#     loaded_model: trained ML model to use for testing
#     full_inp: smap 3d data for testing 
#     time_3d_fmt: smap times in 3d for testing 
#     qlty_flag: quality flag in 3d for testing 
#     time_1d: actual and precise smap acquisition time in 1d array
#     LOOK_BACK: number of measurements in the past to include in the testing 
#     LOOK_BACK_TIME_TEST: actual time in the past to collect smap data for testing 
#     TIME_FORMAT: time format in 'mins', 'hrs' or 'days' 
#     DISP_PLOT: flag for showing plots 
#     SAVE_PLOTS: flag for saving plots """

#     gap_filled = np.empty([full_inp.shape[1], 
#                                     full_inp.shape[2]])   #creates an empty
#     gap_filled  = gap_filled[None, :, :]   #adds an extra dimension; for saving external loop results

#     ground_dum = np.empty([full_inp.shape[1], 
#                                     full_inp.shape[2]])   #creates an empty
#     ground_dum = ground_dum[None, :, :]   #adds an extra dimension; for saving external loop results 

#     qltyFlag_dum = np.empty([full_inp.shape[1], 
#                                     full_inp.shape[2]])   #creates an empty
#     qltyFlag_dum = qltyFlag_dum[None, :, :]   #adds an extra dimension; for saving external loop results                             

#     time_keep_1d = []

#     for i in range(0, 1500):
#     # for i in range(len(full_inp)):
#         print('testing sample i:', i)
#         smap_var_len, time_curr_1d = data_btwTime_idx(full_inp, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
#                             lbk_time = LOOK_BACK_TIME, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, lbk_days_fmt = TIME_FORMAT)

#         qlty_flag_var_len, _ = data_btwTime_idx(qlty_flag, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
#                             lbk_time = LOOK_BACK_TIME, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, lbk_days_fmt = TIME_FORMAT)

#         time_var_len_3d, _ = data_btwTime_idx(time_3d_fmt, time_1d, start_idx=i, lk_bck= LOOK_BACK, 
#                             lbk_time = LOOK_BACK_TIME, lk_fwd = LOOK_FORWARD, lk_fwd_time = LOOK_FORWARD_TIME, lbk_days_fmt = TIME_FORMAT)
        
#         # print('test_res shape:', smap_var_len.shape)
#         # print('qlty_flag_var_len shape:', qlty_flag_var_len.shape)
#         # print('time_var_len_3d:', time_var_len_3d.shape)

#         if len(smap_var_len) != 0:
#             ###########compute time difference
#             time_diff_var_len = time_diff(time_var_len_3d, lk_fwd = LOOK_FORWARD)

#             ######normalize time data
#             time_diff_var_len = time_diff_var_len/LOOK_BACK_TIME  
#             # time_diff_var_len = time_diff_var_len/11                   

#             ######add channels to inputs
#             X_test_full = np.expand_dims(smap_var_len, axis=-1) 
#             X_test_time = np.expand_dims(time_diff_var_len, axis=-1)
#             qlty_flg_temp = np.expand_dims(qlty_flag_var_len, axis=-1) 

#             # print('X_test_full shape:', X_test_full.shape)

#             data_ground, test_res, qlty_flg_temp_slc = singleDay_modelTestwithTime(loaded_model, 
#             X_test_full, X_test_time, qlty_flg_temp, proc_day= i, lk_fwd = LOOK_FORWARD)
#             # print('test_res shape:', test_res.shape)
#             # print('data_ground shape:', data_ground.shape)
#             # print('qlty_flg_temp_slc shape:', qlty_flg_temp_slc.shape)
            

#             ####stack gap-filled smaps and ground-truth smaps
#             gap_filled = np.concatenate((gap_filled, test_res[None, :, :]), axis=0)
#             ground_dum = np.concatenate((ground_dum, data_ground[None, :, :]), axis=0)
#             qltyFlag_dum = np.concatenate((qltyFlag_dum, qlty_flg_temp_slc[None, :, :]), axis=0)

#             ####stack time
#             time_keep_1d.append(time_curr_1d)
#             mae_day, rmse_day = mae_rmse_single_day(data_ground, test_res, qlty_flg_temp_slc) #mae on a specific day without checking valid days
#             if mae_day != np.nan:       #check for days in which the quality flag indicates no valid measurment
#                 mae_day = np.round(mae_day, decimals=4)
#                 data_ground_post, test_res_post = post_proc_smap_res(test_res, data_ground, 
#                                                     qlty_flg_temp_slc)   #apply qlty flags on gap-filled results
                
#                 if plot_operations==True:
#                     plotMap_same_scale(data_ground_post, test_res, plot_title='Results for measurement ' + 
#                             str(i) + '. ' + 'MAE is '+ str(mae_day), sv_plt=SAVE_PLOTS, 
#                             sv_path=save_plot_path, sv_idx=i, custom_cmap=plot_cmap, 
#                             show_plot= DISP_PLOT, lk_bck_days= LOOK_BACK_TIME)  #note that the preprocessed gap-filled result is plotted

#     gap_filled = gap_filled[1:, :, :]     #slice out array excluding the first dummy at index 0
#     ground_dum = ground_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
#     qltyFlag_dum = qltyFlag_dum[1:, :, :]     #slice out array excluding the first dummy at index 0
#     # print('test_res shape:', gap_filled.shape)
#     # print('data_ground shape:', ground_dum.shape)
#     # print('qlty_flg_temp_slc shape:', qltyFlag_dum.shape)

#     time_keep_1d = np.array(time_keep_1d)
#     return ground_dum, gap_filled, qltyFlag_dum, time_keep_1d
















def rotate_stack(data, rot_axis= None):
    """ function: data to be rotated counterclockwise
    rot_axis: axis for which data will be rotated """ 

    if rot_axis==None:
        data_rot = np.rot90(data)
    else:
        data_rot = np.rot90(data, axes= rot_axis)
    return data_rot


def extract_date_save(time_1d):
    """ function: format soil moisture acquisition times by conversion to string and 
    afterwards remove colons and dashes 
    time_1d: soil moisture acquisition times to process """
    tm_lst = []
    for i in range(len(time_1d)):
        # tm_slc = str(time_1d[i, 0])
        # tm_slc_abv = tm_slc[0: 19]
        # tm_lst += [str(tm_slc_abv)]
        if len(time_1d.shape) > 1:
            tm_slc = time_1d[i, 0]
        else:
            tm_slc = time_1d[i]
        tm_slc_abv = np.datetime_as_string(tm_slc, unit='s')
        tm_slc_abv = tm_slc_abv.replace(':', '')      #replace colons with nothing
        tm_slc_abv = tm_slc_abv.replace('-', '')      #replace dashes with nothing
        tm_lst.append(str(tm_slc_abv))
    return tm_lst

def save_geotiffs(arr, time_1d_str, rot= None, colour_map = None, tfile_name = None, dst_file = None):
    """ function: write/save array as geotiff files with tags
    arr: array to save as geotiffs 
    time_1d_str: 1-d array of smap acquisition time as string with colons removed
    rot: flag for rotating data written as geotiff or not 
    colour_map: colour table for displaying saved array 
    tfile_name: tile name
    dst_file: path to write geotiff files """ 
    # the output array should be encoded to byte ot unint16 data type in order to add colormap to the geotiff file
    nan_in = -9999 # the NaN value of input data
    nan_out = 255
    scale_factor = 100
    # encode results using the scale factor
    arr = (arr * float(scale_factor)).round().astype('uint8')
    # reassign NaN values after encoding
    nan_ind = np.where(arr == nan_in)
    arr[nan_ind] = nan_out
    ftile = tfile_name
    geotags = Equi7Tile(ftile).get_tile_geotags()

    if rot == True:
        k_rot=1
    elif rot == False:
        k_rot=0
    # elif rot==None:
    elif rot != True or rot !=False:
        raise RuntimeError('Please indicate if written geotiffs should be rotated or not')


    # time_info = extract_date_save(time_1d)      #retrive time as string for saving geotiffs
    if len(arr.shape)== 3:
        for i in range(len(arr)):
            arr_slc = arr[i, :, :] 
              
            write_tiff(dst_file + tfile_name + '_' + time_1d_str[i] + '_' + '.tif', np.rot90(arr_slc, k=k_rot), tiff_tags=geotags, ct= colour_map)
    elif len(arr.shape)== 2:
        arr_slc = arr
        write_tiff(dst_file + tfile_name + '_' + time_1d_str[i] + '_' + '.tif', np.rot90(arr_slc, k=k_rot), tiff_tags=geotags, ct= colour_map)


def match_test_time(dt, sp_dt= None, tm_diff_thres= None):
    """ function: returns the index and validity flag of the exact or closest time searched in the given time array
    dt: time array in which we perform a search 
    sp_dt: specified time to search 
    tm_diff_thres: allowed threshold during search """ 

    tm_diff_thres = np.timedelta64(tm_diff_thres, 'h')
    # sp_dt = None    #specified date to search
    # tm_diff_thres = None #time difference threshold to consider for valid retrieval 
    for i in range(1, len(dt)):
        # print('i:', i)
        curr_dt = dt[i]
        prev_dt = dt[i - 1]
        if curr_dt == sp_dt:    #best case search in which search precisely exists
            idx = i 
            val_tm_fnd = True                 #flag
            break
        elif curr_dt > sp_dt:
            curr_tm_diff = np.abs(curr_dt - sp_dt)  #get current time
            prev_tm_diff = np.abs(prev_dt - sp_dt)  #get previous time
            if curr_tm_diff < prev_tm_diff:         #check if current time difference is smaller than previous time
                if curr_tm_diff <= tm_diff_thres:   #check that the current time difference meets the specified threshold
                    idx = i
                    val_tm_fnd = True                 #flag to reflect if precise condition is meet
                else:                               #return index as empty if the threshold condition is violated
                    idx = []
                    val_tm_fnd = False                 #flag
            elif prev_tm_diff < curr_tm_diff:       #check if previous time difference is smaller or equal to current time
                if prev_tm_diff < tm_diff_thres:    #check that the previous time difference meets the specified threshold
                    idx = i - 1
                    val_tm_fnd = True                 #flag
                else:                               #return index as empty
                    idx = []
                    val_tm_fnd = False                 #flag
            break
    return idx, val_tm_fnd



def comb_data_from_set(data_1, data_2, data_3, data_4, time_1d_1, time_1d_2, time_1d_3, time_1d_4, rot=True,
        y_dim = None, x_dim= None):
    """ function: returns combine soil mositure measurements/images and the corresponding acquisition times for neighbouring tiles by aligning them based on acquisition times 
    data_1: smap stack 1
    data_2: smap stack 2
    data_3: smap stack 3 
    data_4: smap stack 4 
    time_1d_1: smap aquisition time for data_1
    time_1d_2: smap aquisition time for data_2
    time_1d_3: smap aquisition time for data_3
    time_1d_4: smap aquisition time for data_4
    rot: flag for rotating stack of single measurement/image by 90 degrees 
    y_dim: the expected new y-axis dismension of the combined smap 
    x_dim: the expected new x-axis dismension of the combined smap """ 

    time_dum = []
    dum_all = np.empty([y_dim, x_dim])   #creates an empty
    dum_all  = dum_all[None, :, :]   #adds an extra dimension; for saving external loop results

    ln_min = min(data_1.shape[0], data_2.shape[0], data_3.shape[0], data_4.shape[0])    #find the stack with minimum length

    # print('data_1 shape:', len(data_1.shape))
    for i in range(120):
    # for i in range(ln_min):

        idx_out_1, val_time_flag_1 = match_test_time(time_1d_1, sp_dt= time_1d_1[i], tm_diff_thres= 2)
        idx_out_2, val_time_flag_2 = match_test_time(time_1d_2, sp_dt= time_1d_1[i], tm_diff_thres= 2)
        idx_out_3, val_time_flag_3 = match_test_time(time_1d_3, sp_dt= time_1d_1[i], tm_diff_thres= 2)
        idx_out_4, val_time_flag_4 = match_test_time(time_1d_4, sp_dt= time_1d_1[i], tm_diff_thres= 2)

        if len(data_1.shape) ==3:         #works for stacked measurements
            # print('i excuted:', i)
            if val_time_flag_1==True and val_time_flag_2== True and val_time_flag_3== True and val_time_flag_4== True:
                if rot== True:
                    data_1_rot = rotate_stack(data_1, rot_axis= (1, 2))
                    data_2_rot = rotate_stack(data_2, rot_axis= (1, 2))
                    data_3_rot = rotate_stack(data_3, rot_axis= (1, 2))
                    data_4_rot = rotate_stack(data_4, rot_axis= (1, 2))

                    data_1_rot = data_1_rot[idx_out_1, :, :]
                    data_2_rot = data_2_rot[idx_out_2, :, :]
                    data_3_rot = data_3_rot[idx_out_3, :, :]
                    data_4_rot = data_4_rot[idx_out_4, :, :]
                else:
                    data_1_rot = data_1[idx_out_1, :, :]
                    data_2_rot = data_2[idx_out_2, :, :]
                    data_3_rot = data_3[idx_out_3, :, :]
                    data_4_rot = data_4[idx_out_4, :, :]

        elif len(data_1.shape)==2:            #works for individual measurements
            if rot== True:
                data_1_rot = rotate_stack(data_1)
                data_2_rot = rotate_stack(data_2)
                data_3_rot = rotate_stack(data_3)
                data_4_rot = rotate_stack(data_4)
            else:
                data_1_rot = data_1
                data_2_rot = data_2
                data_3_rot = data_3
                data_4_rot = data_4

        if val_time_flag_1==True and val_time_flag_2== True and val_time_flag_3== True and val_time_flag_4== True:
            data_cat_col_1 = np.concatenate((data_1_rot, data_2_rot), axis=1)
            data_cat_col_2 = np.concatenate((data_3_rot, data_4_rot), axis=1)
            data_cat_all = np.concatenate((data_cat_col_1, data_cat_col_2), axis=0)
            time_dum += [time_1d_1[i]]      #store valid/aligned smap acquisition times

            dum_all = np.concatenate((dum_all, data_cat_all[None, :, :]), axis=0)
    if len(dum_all) > 1:
        dum_all = dum_all[1:, :, :]     #slice out array excluding the first dummy at index 0
    else:
        dum_all = []
    return dum_all, np.array(time_dum)


def apply_qlty_flags_smap(smap_data, qlty_flag, fill_val= -0.01):
    """ function: returns smap data with given quality flag and fill values applied
    smap_data: smap data to process 
    qlty_flag: quality flags 
    fill_val: fill value for invalid smap pixels """ 

    flg = qlty_flag.copy()
    smap = smap_data.copy()
    # flg = np.nan_to_num(flg, copy=True, nan= 0)         #replace nan with fill value
    flg = flg * -1.0                #make non-zero values (i.e. bad values) negatives
    flg[flg < 0] = -9999.0        #replace non-zero values (i.e. flag for bad values) with fill values for bad values
    flg[flg == 0] = 1        #replace zeros (i.e. flag for good values) values with ones
    flg[flg == -9999.0] = 0        #replace fill values (i.e. flag for bad values) values with ones
    for i in range(len(flg)):
        # print('i:', i)
        flg_slc = flg[i, :, :]
        smap_slc = smap[i, :, :]
        smap[i, :, :] = np.multiply(smap_slc, flg_slc)      #takes care of all missing and invalid sm values
    smap[smap== 0] = fill_val 
    # smap = np.nan_to_num(smap, copy=True, nan= -0.01) 
    return smap, flg


def insert_cygnns_in_smap(smap, cygnss, thresh=None):
    """ function: this function inserts cygnss pixel values in smap data if the cygnns pixel value is greater than the 
    corresponding smap pixel value 
    smap: smap data in 2-d 
    cygnss: cygnss data in 2-d 
    thresh: threshold for determining insertion """

    data = smap.copy()
    y, x = smap.shape[0], smap.shape[1]
    rep_coord = []
    rep_cnt= 0

    for j in range(0, y):   #read y-axis
        for i in range (0, x):  #read x-axis
            smap_pix = smap[j, i]  #read smap pixel value
            cygnss_pix = cygnss[j, i]  #read cygnss pixel value
            if cygnss_pix > smap_pix: 
                if cygnss_pix > thresh:   
                    data[j, i] = cygnss_pix    #insert cgynss in the copy of smap data
                    rep_coord.append((j, i))       #append coordinates of insertion
                    rep_cnt += 1
            else:
                pass
    return data, rep_coord, rep_cnt


def insert_cygnns_in_smap_dataset(smap, cygnss, thresh=None, abv_thresh=None):
    """ function: this function inserts cygnss pixel values in smap data if the cygnns pixel value is greater than the 
    corresponding smap pixel value 
    smap: smap data in 2-d 
    cygnss: cygnss data in 2-d """ 
    
    data = smap.copy()
    y, x = smap.shape[1], smap.shape[2]
    rep_cnt = 0
    rep_coord = []
    rep_coord_ch = []

    for k in range(len(smap)):
        # smap_slc = smap[k, :, :]    #extracts 2d slice
        # cygnss_slc = cygnss[k, :, :]    #extracts 2d slice
        
        for j in range(0, y):   #read y-axis
            for i in range (0, x):  #read x-axis
                smap_pix = smap[k, j, i]  #read smap pixel value
                cygnss_pix = cygnss[k, j, i]  #read cygnss pixel value
                if cygnss_pix > smap_pix:
                    # if cygnss_pix > thresh:
                    if cygnss_pix - smap_pix > thresh and cygnss_pix > abv_thresh:
                        data[k, j, i] = cygnss_pix    #insert cgynss in the copy of smap data
                        # data[k, j, i] = (cygnss_pix + 0.5 * smap_pix)/2
                        rep_coord_ch.append((j, i))
                        rep_cnt += 1
                else:
                    pass
        rep_coord.append((k, rep_coord_ch))
    return data, rep_coord, rep_cnt




def surrounding_tiles(tile_name = None):
    """This function returns a list of the 8 surrounding tiles in the order: NW,N,NE,W,E,SW,S,SE. Input is a string in
    the format NA6000M_E078N024T6."""
    if tile_name != None:
        east = int(tile_name[9:12])
        north = int(tile_name[13:16])
        ul_tile_name = tile_name[:9] + str(east - 6).zfill(3) + "N" + str(north + 6).zfill(3) + tile_name[-2:]
        uc_tile_name = tile_name[:9] + str(east).zfill(3) + "N" + str(north + 6).zfill(3) + tile_name[-2:]
        ur_tile_name = tile_name[:9] + str(east + 6).zfill(3) + "N" + str(north + 6).zfill(3) + tile_name[-2:]
        l_tile_name = tile_name[:9] + str(east - 6).zfill(3) + "N" + str(north).zfill(3) + tile_name[-2:]
        r_tile_name = tile_name[:9] + str(east + 6).zfill(3) + "N" + str(north).zfill(3) + tile_name[-2:]
        ll_tile_name = tile_name[:9] + str(east - 6).zfill(3) + "N" + str(north - 6).zfill(3) + tile_name[-2:]
        lc_tile_name = tile_name[:9] + str(east).zfill(3) + "N" + str(north - 6).zfill(3) + tile_name[-2:]
        lr_tile_name = tile_name[:9] + str(east + 6).zfill(3) + "N" + str(north - 6).zfill(3) + tile_name[-2:]
        tiles = [ul_tile_name,uc_tile_name,ur_tile_name,l_tile_name,r_tile_name,ll_tile_name,lc_tile_name,lr_tile_name]
        return tiles


def comb_stack(tile_name,surrounding_tiles, file_path):
    sm = np.load(os.path.join(file_path, tile_name + "_comb_merged_2.npy"))
    date = np.load(os.path.join(file_path, tile_name + "_local_time_2.npy"))
    mask = np.load(os.path.join(file_path, tile_name + "_mask_2.npy"))

    # sm = np.load(os.path.join(file_path, tile_name + "_comb_gf.npy"))
    # date = np.load(os.path.join(file_path, tile_name + "_local_time.npy"))
    # mask = np.load(os.path.join(file_path, tile_name + "_mask.npy"))

    sm_stack = np.zeros((sm.shape[0], 300, 300)) - 9999
    date_stack = np.zeros((date.shape[0], 300, 300)) - 9999
    mask_stack = np.zeros((date.shape[0], 300, 300))
    sm_stack[sm_stack == -9999] = np.nan
    date_stack[date_stack == -9999] = np.nan
    sm_stack[:,100:200,100:200] = sm
    date_stack[:,100:200,100:200] = date
    mask_stack[:,100:200,100:200] = mask
    pos = np.asarray(
        [[0, 100, 0, 100], [0, 100, 100, 200], [0, 100, 200, 300], [100, 200, 0, 100],
         [100, 200, 200, 300], [200, 300, 0, 100], [200, 300, 100, 200], [200, 300, 200, 300]])
    for i in range(len(surrounding_tiles)):
        if os.path.exists(os.path.join(file_path, surrounding_tiles[i] + "_comb_gf.npy")):
            x = pos[i, 2]
            x2 = pos[i, 3]
            y = pos[i, 0]
            y2 = pos[i, 1]
            sm_stack[:, y:y2, x:x2] = np.load(os.path.join(file_path, surrounding_tiles[i] + "_comb_merged_2.npy"))
            date_stack[:, y:y2, x:x2] = np.load(os.path.join(file_path, surrounding_tiles[i]+ "_local_time_2.npy"))
            mask_stack[:, y:y2, x:x2] = np.load(os.path.join(file_path, surrounding_tiles[i] + "_mask_2.npy"))

            # sm_stack[:, y:y2, x:x2] = np.load(os.path.join(file_path, surrounding_tiles[i] + "_comb_gf.npy"))
            # date_stack[:, y:y2, x:x2] = np.load(os.path.join(file_path, surrounding_tiles[i]+ "_local_time.npy"))
            # mask_stack[:, y:y2, x:x2] = np.load(os.path.join(file_path, surrounding_tiles[i] + "_mask.npy"))
        else:
            pass
    # sm_stack = sm_stack[:, 50:250, 50:250]
    # date_stack = date_stack[:, 50:250, 50:250]
    # mask_stack = mask_stack[:, 50:250, 50:250]

    sm_stack = sm_stack[:, 75:225, 75:225]
    date_stack = date_stack[:, 75:225, 75:225]
    mask_stack = mask_stack[:, 75:225, 75:225]
    return sm_stack, date_stack, mask_stack

