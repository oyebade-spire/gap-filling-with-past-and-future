import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from data_utilities import temporalize_test_now, add_gaps_data, get_valid_days, data_btwTime_idx, data_bwtTime_dataset
from data_utilities import mae_rmse_single_day, mae_dataset, get_valid_days_full, valid_sm_single_day, pix_grd_vld_time
from data_utilities import singleDay_modelTest, pixels_timeSeries, add_gap_day, post_proc_smap_res, format_time
from data_utilities import singleDay_modelTestwithTime, time_diff, var_len_testModel
from data_utilities_nc import add_spatial_gaps, add_temporal_gaps
from plot_utilities import plot_data, plot_train_hist, plotMap_same_scale, plot_scatter, plot_abs_error, create_colormap
from train_utilities import lr_schedule
import matplotlib.pyplot as plt
import datetime
import build_model

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

##############path for saving gap-filled results with natural gaps
path_save_fig = "/home/ubuntu/Oyebade/gap_filling_fut/results/gap_filled_smaps/"

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
LOOK_FORWARD = 3     #number of measurements/images
LOOK_FORWARD_TIME = 5   #maximum real time in the future from which training data is extracted; values are in 'hrs' or 'days'

TIME_FORMAT = 'days'  #time format in 'mins', 'hrs' or 'days'

##################load data
test_full_data = np.load('/home/ubuntu/Oyebade/gap_filling/proj_data/full_processedSmap/NA/SMAP_FULL_NA6000M_E078N024T6.npz')
# test_full_data = np.load('/home/ubuntu/Oyebade/gap_filling/proj_data/smap_data_full.npz')
full_inp = test_full_data['full_inp']
# full_mask = test_full_data['full_mask']
qlty_flag = test_full_data['qlty_flag']
surf_flag = test_full_data['surf_flag']
time_3d = test_full_data['time_3d']
time_1d = test_full_data['time_1d']

# full_inp = np.nan_to_num(full_inp, copy=True, nan= -0.01)         #replace nan with fill value
full_inp[full_inp < -1000] = -0.01         #replace -9999.0 values with -1

#################change time format to hours
time_3d_fmt = format_time(time_3d, fmt= TIME_FORMAT)

################load the trained ML model
loaded_model = load_model('/home/ubuntu/Oyebade/gap_filling_fut/models/gfModel_0.85M_LB_7_0.01.h5')
# loaded_model = load_model('/home/ubuntu/Oyebade/gap_filling_fut/models/MAE_GOOD_gfModel_0.85M_LB_7_0.01.h5')
# loaded_model = load_model('/home/ubuntu/Oyebade/gap_filling_fut/models/GF_0.85M_LB_7_0.01.h5')
loaded_model.summary()

###################testing with natural gaps in smap data
data_ground, test_res_full, qlty_flg_temp_slc, time_keep_1d = var_len_testModel(loaded_model, full_inp, time_3d_fmt, qlty_flag, time_1d, 
            LOOK_BACK, LOOK_BACK_TIME, TIME_FORMAT, LOOK_FORWARD, LOOK_FORWARD_TIME, 
            plot_operations= True, plot_cmap= CUST_CMAP, DISP_PLOT= False, SAVE_PLOTS= True, save_plot_path= path_save_fig)

###################testing with artificial gaps in smap data
if test_with_artificial_gap== True:
    gap_filled_results = np.load(smapData_art_gap_path + SmapData_art_gap + '.npz')
    smap_gap_30 = gap_filled_results['smap_gap_30']
    # smap_gap_50 = gap_filled_results['smap_gap_50']
    # qlty_flag = gap_filled_results['qlty_flag']
    # time_smap = gap_filled_results['time_smap']

data_ground, test_res_full, qlty_flg_temp_slc, time_keep_1d = var_len_testModel(loaded_model, smap_gap_30, time_3d_fmt, qlty_flag, time_1d, LOOK_BACK, LOOK_BACK_TIME_TEST, TIME_FORMAT,
            plot_operations= True, plot_cmap= CUST_CMAP, DISP_PLOT= True, SAVE_PLOTS= True, save_plot_path= path_save_fig)

###################save gap-filled smap results as arrays depending the flag
if save_array_test_results:
    print('excuted')
    np.savez(gapFilled_smapData_path + full_SmapData_name, gap_filled_smap= test_res_full, time_smap= time_keep_1d) 
    gap_filled_results = np.load(gapFilled_smapData_path + full_SmapData_name + '.npz')
    gap_filled_smap = gap_filled_results['gap_filled_smap']
    time_smap = gap_filled_results['time_smap']

##################plot results as time series for a specified day
idx = 9
data_ground_vld, cnt = valid_sm_single_day(data_ground[idx, :, :], qlty_flg_temp_slc[idx, :, :])  #get valid entries in data using quality flags
test_res_vld, _ = valid_sm_single_day(test_res_full[idx, :, :], qlty_flg_temp_slc[idx, :, :])

plot_scatter(data_ground_vld, test_res_vld, lb_1="ground-truth SMAP", lb_2="gap-filled SMAP with complete look-back", 
    y_lbl='Soil moisture', x_lbl='Pixel', plt_title='Reconstruction of valid soil moisture for day' + ' ' + str(idx))

##############plot for a single pixel, gap-filled sm as time-series over time
pixel_idx = 3
pix_grnd_vld, pix_res_slc = pixels_timeSeries(data_ground, test_res_full, 
        qlty_flg_temp_slc, pix_idx=pixel_idx)

plot_scatter(pix_grnd_vld, pix_res_slc, lb_1="ground-truth SMAP", 
    lb_2="gap-filled SMAP with complete look-back", y_lbl='Soil moisture', 
    x_lbl='Day', plt_title='Reconstruction of valid soil moisture for pixel' + ' ' + str(pixel_idx))



###############################################################################################################################################
####create atificial spatial and temporal gaps in original dataset
