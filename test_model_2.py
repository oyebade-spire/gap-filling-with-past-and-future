from osgeo import ogr, osr
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from data_utilities import  extract_date_save, surrounding_tiles, comb_stack, broadcast_unique_time
from data_utilities import save_geotiffs, var_len_testModel_fut, static_qlty_flag_mask, get_mask
from data_utilities import time_str_comb, get_unique_file_names, format_time, retrieve_tm_utc
from data_utilities_nc import empty_folder, s3_transfer
from plot_utilities import plot_data, create_colormap
import matplotlib.pyplot as plt
import build_model
from pygnssr.common.utils.gdalport import gen_gdal_ct
from netCDF4 import Dataset, num2date
import pandas as pd


#################writing plots as geotiff files
colormap_file =r"/home/ubuntu/Oyebade/gap_filling/proj_data/colourtables/ct_cssm.ct"
ct = gen_gdal_ct(colormap_file)
save_geotiff_path = '/home/ubuntu/Oyebade/gap_filling_fut/results/smap/gap_filled_geotiffs/'
ground_geotiff_path = '/home/ubuntu/Oyebade/gap_filling_fut/results/smap/ground_truth_geotiffs/'
pix_qlty_path = '/home/ubuntu/Oyebade/gap_filling_fut/results/smap/pix_qltyFlgs/'

##############path for saving gap-filled results with natural gaps
path_save_fig = "/home/ubuntu/Oyebade/gap_filling_fut/results/smap/pngs/"

##############for allowing the growth of gpu resources
gpus = tf.config.list_physical_devices('GPU') 
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('gpus being used:', len(gpus))

##############Optimize tensor layouts. e.g. This will try to use NCHW layout on GPU which is faster
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})   #keeps layout as default NHWC

save_all_gapFilled_plots= True     #flag used for saving all gap-filled results as plots
test_with_artificial_gap = False     #flag used for determining, whether to test with artificial gaps
save_smapData_art_gap = False        #flag for saving smap with artificial gap
#############increase plot resolution
plt.rcParams['figure.dpi'] = 300

##############get cutomized colour map for plotting results
custom_cmap_path = "/home/ubuntu/Oyebade/gap_filling_fut/proj_data/cssm_1.txt"  #path for cutomized colour map txt file for plotting results
CUST_CMAP = create_colormap(custom_cmap_path)

################flag and path for saving gap-filled smap in array form
save_array_test_results = False    #flag to save gap-filled smap as arrays
gapFilled_smapData_path = '/home/ubuntu/Oyebade/gap_filling_fut/proj_data/smap_gapFilled_arrays/'   #path for saving gap-filled smap results in array form
full_SmapData_name = 'gapFilled_SMAP_L3_NA6000M_E078N024T6'           #name used for saving gap-filled smaps in array form

###############testing with artificial gaps. Path from which to load validation data in array form
smapData_art_gap_path = '/home/ubuntu/Oyebade/gap_filling/results/smap_ext_valid/'  #path for saving results artificial gaps
SmapData_art_gap = 'smap_with_art_gaps'   #path for saving smap with added artificial gaps

###############testing hyper-parameters for trained ML model
BATCH_SZ_TEST = 64   #testing batch size
LOOK_BACK = 7        #number of measurements to look back from the current measurement
LOOK_BACK_TIME = 9     #maximum number of days from which training data is extracted; values are in 'hrs' or 'days'
LOOK_FORWARD = 1     #number of measurements/images
LOOK_FORWARD_TIME = 2   #maximum real time in the future from which training data is extracted; values are in 'hrs' or 'days'

TIME_FORMAT = 'days'  #time format in 'mins', 'hrs' or 'days'


# data_folder = 'NA6000M_LT_10px_window/'
data_folder = 'OC6000M_LT_10px_window/'
# data_folder = 'SA6000M_LT_10px_window/'
# data_folder = 'AS6000M_LT_10px_window/'


# local_data_root_path = '/home/ubuntu/Oyebade/gap_filling/proj_data/Dennis_ml/'
local_data_root_path = '/home/ubuntu/datapool/internal/ei-land_working_dir/dennis/stat_gap_fill/output/'

################load the trained ML model
ml_model_path = '/home/ubuntu/Oyebade/gap_filling_fut/models/gf_fut_NA6000M_E078N024T6_den.h5'
loaded_model = load_model(ml_model_path)
loaded_model.summary()

#############get alist of unique file names in the specified directory to process
unique_file_names_lst = get_unique_file_names(local_data_root_path, data_folder)

gap_filled_fld = '/gap_filled_geotiffs/'      #path on local machine to processed gap-filled geotiffs
ground_truth_fld = '/ground_truth_geotiffs/'      #path on local machine processed to ground-truths
pix_qlty_fld = '/pix_qltyFlgs/'      #path on local machine processed to xomputed pix quality

local_res_fld = '/home/ubuntu/Oyebade/gap_filling_fut/results/smap'

# s3_parent_root_fld = 's3://gnss-datapool/internal/ei-land_working_dir/oyebade/results_fut_FromDennisData_withBuffer/'
s3_parent_root_fld = 's3://gnss-datapool/internal/ei-land_working_dir/oyebade/results_fut_FromDennisData_withBuffer_rough/'

# s3_folders_to_save = [png_fld, gap_filled_fld, ground_truth_fld, pix_qlty_fld, arr_filled_fld]  #similar folders are used on the local machine
s3_folders_to_save = [gap_filled_fld, ground_truth_fld, pix_qlty_fld]  #similar folders are used on the local machine



def gap_fill_tiles(unique_file_names_lst, local_res_fld_path, s3_parent_root_path, s3_folders_to_save, save_gf_array=False, comp_pix_qlty=True, write_to_s3_buc=True):
    """ function: gap-fills all the tiles in the specified folder (i.e. 'data_folder'), deletes all current contents in the specified 
    saving path on local machine, writes to local storage, then uploads the results to s3 bucket 
    local_data_root_path: parent directory to 'data_folder'
    data_folder: folder where lies the actual data of interest to process
    s3_folders_to_save: list of folders to create on s3 bucket and then write results into """ 

    print('number of tiles to process:', len(unique_file_names_lst))
    
    for i in range(len(unique_file_names_lst)):

        print('processing tile:', unique_file_names_lst[i])

        tile_name = unique_file_names_lst[i]        #tile name
        cont_name = tile_name.split('_')[0]  #extract continent names

        ##########load surrounding tiles
        surrounding_tiles_names = surrounding_tiles(tile_name = tile_name)

        ##########combine tiles
        data_path = local_data_root_path + data_folder
        smap_data, smap_date, smap_flag = comb_stack(tile_name, surrounding_tiles_names, data_path) #load data here

        # smap_data = np.load(local_data_root_path + data_folder + tile_name + '_comb_gf.npy')
        # smap_date = np.load(local_data_root_path + data_folder + tile_name + '_local_time.npy')
        # smap_flag = np.load(local_data_root_path + data_folder + tile_name + '_mask.npy')

        # png_fld = '/' + tile_name + s3_folders_to_save[0]        #path on local machine to processed pngs
        gap_filled_fld = '/' + tile_name + s3_folders_to_save[0]      #path on local machine to processed gap-filled geotiffs
        ground_truth_fld = '/' + tile_name + s3_folders_to_save[1]     #path on local machine processed to ground-truths
        pix_qlty_fld = '/' + tile_name + s3_folders_to_save[2]           #path on local machine to save the pix quality computed
        # arr_res_fld = '/' + tile_name + s3_folders_to_save[4]           #path on local machine to save the gap-filled arrays
        
        s3_folders_results = [gap_filled_fld, ground_truth_fld, pix_qlty_fld]

        ############format Dennis'data to match my code expectations for testing
        smap_time_rev = broadcast_unique_time(smap_date)

        ##############smap input
        if len(smap_flag) !=0:     #apply merged data flag if available according to cygnss
            smap_data[smap_flag == 0] = -0.01  
        smap_data = np.nan_to_num(smap_data, copy=True, nan= -0.01, posinf=None, neginf=None)


        ##########update mask for soil moisture data
        smap_flag = get_mask(smap_data, fill_val=-0.01)

        #########time input
        time_3d_fmt = format_time(smap_time_rev, fmt_curr = 'secs', fmt_to= TIME_FORMAT)

        #########real dates and times
        smap_date_slc = smap_date[:, 25: 125, 25: 125]      #slice out the actual tile times
        time_min_1d, time_max_1d  = retrieve_tm_utc(smap_date_slc)

        #convert times to string and remove colons and dashes
        time_min_1d_str = extract_date_save(time_min_1d)    
        time_max_1d_str = extract_date_save(time_max_1d)   

        tm_str_comb = time_str_comb(time_min_1d_str, time_max_1d_str)

        #############################compute static quality flag using the entire stack
        static_qlty_flg = static_qlty_flag_mask(smap_flag)

        ####################local machine paths to write pngs/geotiffs/arrays
        local_folder_0 = local_res_fld_path + s3_folders_to_save[0]
        local_folder_1 = local_res_fld_path + s3_folders_to_save[1]
        local_folder_2 = local_res_fld_path + s3_folders_to_save[2]

        ##########################empty destination paths
        empty_folder(local_folder_0)    #path to the gap-filled data
        empty_folder(local_folder_1)    #path to the ground-truth data
        empty_folder(local_folder_2)    #path to the pixel quality data

        ###################testing with natural gaps in smap data
        test_res_full, pix_qlty_2d_dum = var_len_testModel_fut(loaded_model, smap_data, time_3d_fmt, 
        smap_flag, static_qlty_flg, time_max_1d, LOOK_BACK, LOOK_BACK_TIME, TIME_FORMAT, LOOK_FORWARD, LOOK_FORWARD_TIME,
        rot_stack=False, applyStatFlag=False, time_1d_str_form= tm_str_comb, tile_name= tile_name, comp_pixSeriesQlty= comp_pix_qlty, 
        plot_operations= False, plot_cmap= CUST_CMAP, DISP_PLOT= True, SAVE_PLOTS= True, save_plot_path= path_save_fig)

         ###################get length of retrieved gap-filled results
        len_res = len(test_res_full)
        
        ###########slice out centre aspect of interest
        test_res_full_wr = test_res_full[:, 25: 125, 25: 125]  #take 25:125 when smap input is cut 75:175
        smap_data_wr = smap_data[:len_res, 25: 125, 25: 125]   #take 25:125 when smap input is cut 75:175
        # test_res_full_wr = test_res_full[:, 50: 150, 50: 150]  #take 50:150 when smap input is cut 50:250
        # smap_data_wr = smap_data[:len_res, 50: 150, 50: 150]   #take 50:150 when smap input is cut 50:250
        # pix_qlty_2d_dum = pix_qlty_2d_dum                   #this is already usually always 100 by 100 
        
        #####################save results as geotiff files
        save_geotiffs(test_res_full_wr, tm_str_comb, rot= False, colour_map = ct, tfile_name = tile_name, dst_file = save_geotiff_path)   #save gap-filled results as geotiff files
        save_geotiffs(smap_data_wr, tm_str_comb, rot= False, colour_map = ct, tfile_name = tile_name, dst_file = ground_geotiff_path) #save ground-truth as geotiff files
        if comp_pix_qlty:
            save_geotiffs(pix_qlty_2d_dum, tm_str_comb, rot= False, colour_map = ct, tfile_name = tile_name, dst_file = pix_qlty_path) #save ground-truth as geotiff files
  
        if write_to_s3_buc:
            s3_remote_folder_0 = s3_parent_root_path + cont_name + s3_folders_results[0]
            s3_remote_folder_1 = s3_parent_root_path + cont_name + s3_folders_results[1]
            # s3_transfer(local_folder_1, s3_remote_folder_1, proc_data_type='folder', trans_type= 'upload')
            s3_transfer(local_folder_0, s3_remote_folder_0, proc_data_type='folder', trans_type= 'upload')
            s3_transfer(local_folder_1, s3_remote_folder_1, proc_data_type='folder', trans_type= 'upload')
            if comp_pix_qlty:
                s3_remote_folder_2 = s3_parent_root_path + cont_name + s3_folders_results[2]
                s3_transfer(local_folder_2, s3_remote_folder_2, proc_data_type='folder', trans_type= 'upload')
        
    return print('finshed processing all tiles in the specified folder')


# lst_proc = ['NA6000M_E072N024T6', 'NA6000M_E078N024T6', 
#     'NA6000M_E072N018T6', 'NA6000M_E078N018T6']
lst_proc = ['OC6000M_E078N060T6', 'OC6000M_E084N060T6', 
    'OC6000M_E078N054T6', 'OC6000M_E084N054T6']
gap_fill_tiles(lst_proc, local_res_fld, s3_parent_root_fld, s3_folders_to_save, 
    save_gf_array=False , comp_pix_qlty=True, write_to_s3_buc= True)
# gap_fill_tiles(unique_file_names_lst, local_res_fld, s3_parent_root_fld, s3_folders_to_save, 
#     save_gf_array=False , comp_pix_qlty=True, write_to_s3_buc= True)


