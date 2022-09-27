from osgeo import ogr, osr
import tensorflow as tf
import os
import math
import numpy as np
import numpy
from tensorflow import keras
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint
import tensorflow_datasets as tfds
from build_model import Model_cons
from data_utilities import DataSequence_train, DataSequence_test, get_filenames, temporalize, temporalize_now
from data_utilities import temporalize_sm_tm_fut, format_time, split_trainTest_data
from plot_utilities import plot_data, plot_train_hist
from train_utilities import lr_schedule
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import gc

####################for allowing the growth of gpu resources
gpus = tf.config.list_physical_devices('GPU') 
if len(gpus) > 0:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('gpus being used:', len(gpus))

####################Optimize tensor layouts. e.g. This will try to use NCHW layout on GPU which is faster
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})   #keeps layout as default NHWC

####################variables
REG_WEIGHT = 0
REG_WEIGHT_REC = 0
EPOCHS = 6   #training epochs
BATCH_SZ = 16     #training batch size
BATCH_SZ_TEST = 64    #testing batch size
VAL_FREQ = 1    #simulate validation data after this epochs
IMG_ROWS, IMG_COLS = None, None        #use the relation or set as None
IMG_CHANNELS       = 1  #number of input channels
OUT_CHANNELS       = 1  #number of output channels
IMG_INPUT = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)  #input image shape
NORM_LAYS = True
LOOK_BACK = 6       #number of measurements/images
LOOK_FORWARD = 1    #number of measurements/images
MAX_LOOK_BACK_TIME = 10     #maximum real time in the past from which training data is extracted; values are in 'hrs' or 'days'
MAX_LOOK_FORWARD_TIME = 2   #maximum real time in the future from which training data is extracted; values are in 'hrs' or 'days'
MODEL_WIDTH = 1     #multiplicative factor for model filters
LEAKY_RELU = 0  #leaky factor for leaky relu activation function
MODEL_CHKPOINT = False  #flag that reflects, whether to checkpoint model
MODEL_LOSS = 'mae'      #model loss function; values are 'mae' or 'mse
TIME_TRAIN_FORMAT_CURR = 'secs'  #the current format in which time is represented; in 'secs' or 'nanosecs'
MAX_TIME_TRAIN_FORMAT_TO = 'days'    #time format to covert to in 'mins', 'hrs' or 'days'

CONV_LSTM_IMG_INPUT = (None, None, None, IMG_CHANNELS)   #input shape for SMAP
CONV_LSTM_TM_INPUT = (None, None, None, IMG_CHANNELS)    #input shape for time
SAVE_TRAINED_MODEL = True      #flag for saving trained model

#################paths and saving names for variables
checkpoint_filepath="/home/ubuntu/Oyebade/gap_filling_fut/models/checkpoint_gfModel_0.85M_LB_7_0.01.h5"   #model checkpoint path
model_name = '/home/ubuntu/Oyebade/gap_filling_fut/models/gf_fut_NA6000M_E078N024T6_den.h5'   #path and model name for saving trained model
save_model_loss_path = '/home/ubuntu/Oyebade/gap_filling_fut/results/'    #path to save observed metrics evolution plot for trained model
trained_model_train_curve = 'model training for gap-filling with conv-lstm with future data.png'  #name of used for saving plotted curves for trained model

###################read training data
data_all = np.load('/home/ubuntu/Oyebade/gap_filling/proj_data/smap/trainData/NA/SMAP_PATCH_NA6000M_E078N024T6_Den_2.npz')
# data_all = np.load('/home/ubuntu/Oyebade/gap_filling/proj_data/trainData/NA/SMAP_PATCH_NA6000M_E078N024T6.npz')
# data_all = np.load('/home/ubuntu/Oyebade/gap_filling/proj_data/smap_data_coord_sp_tem_seq1.npz')
data_inp = data_all['inp']
data_out = data_all['out']
time = data_all['time']
coord = data_all['coord']

##################optional training data slicing; can alleviate memory problems
data_lm = 100000    #used for limiting data for training due to memory issues
data_inp = data_inp[:data_lm, :, :]
data_out = data_out[:data_lm, :, :]
time = time[:data_lm, :, :]

###################format time difference in 'mins', 'hrs', 'days'
time = format_time(time, fmt_curr= TIME_TRAIN_FORMAT_CURR, 
    fmt_to= MAX_TIME_TRAIN_FORMAT_TO)


###################Add a channel dimension since the images are grayscale.
data_inp = np.expand_dims(data_inp, axis=-1)
time = np.expand_dims(time, axis=-1)
data_out = np.expand_dims(data_out, axis=-1)

##################Generated training sequences for use in the model.
X, z, y, tm_diff = temporalize_sm_tm_fut(data_inp, time, data_out, 
                            lookback=LOOK_BACK, max_lk_bck_time= MAX_LOOK_BACK_TIME, lookforward=LOOK_FORWARD, 
                            max_lk_fwd_time= MAX_LOOK_FORWARD_TIME)	#using only sm patches as input

################normalize time data in hours, since z is in hours
z = z/MAX_LOOK_BACK_TIME 

################garbage collection
# del data_inp, data_out, time, coord
# gc.collect()

""" 
ConvLSTM2D layer, which will accept inputs of shape (batch_size, num_frames, 
width, height, channels) """

#################import conv lstm model
# if len(gpus)> 1:
#     ######setup multi-gpu training
#     strategy = tf.distribute.MirroredStrategy()
#     print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#     with strategy.scope():
#         # ml_class = Model_cons(CONV_LSTM_IMG_INPUT, CONV_LSTM_TM_INPUT, use_norm= NORM_LAYS)
#         # model = ml_class.gf_CAE_LSTM_model(width=MODEL_WIDTH, l_val= LEAKY_RELU)
#         ml_class = Model_cons(CONV_LSTM_IMG_INPUT, CONV_LSTM_TM_INPUT, use_norm= NORM_LAYS)
#         model = ml_class.gf_CAE_LSTM_tm_model(width=MODEL_WIDTH, l_val= LEAKY_RELU)     #model with soil moisture and time inputs
# else:   #use single gpu training setup
#     # ml_class = Model_cons(CONV_LSTM_IMG_INPUT, CONV_LSTM_TM_INPUT, use_norm= NORM_LAYS)
#     # model = ml_class.gf_CAE_LSTM_model(width=MODEL_WIDTH, l_val= LEAKY_RELU)
#     ml_class = Model_cons(CONV_LSTM_IMG_INPUT, CONV_LSTM_TM_INPUT, use_norm= NORM_LAYS)
#     model = ml_class.gf_CAE_LSTM_tm_model(width=MODEL_WIDTH, l_val= LEAKY_RELU)     #model with soil moisture and time inputs


if len(gpus)> 1:
    ######setup multi-gpu training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        ml_class = Model_cons(CONV_LSTM_IMG_INPUT, CONV_LSTM_TM_INPUT, use_norm= NORM_LAYS)
        model = ml_class.gf_CAE_LSTM_tm_model(width=MODEL_WIDTH, l_val= LEAKY_RELU, w_dec=REG_WEIGHT, w_dec_rec=REG_WEIGHT_REC)     #model with soil moisture and time inputs
        # model = ml_class.gf_dia_CAE_LSTM_tm_model(width=MODEL_WIDTH, l_val= LEAKY_RELU, w_dec=REG_WEIGHT)   #with dialated convolution
        # model = ml_class.gf_CAE_LSTM_tm_large_model(width=MODEL_WIDTH, l_val= LEAKY_RELU)     #model with soil moisture and time inputs      
else:   #use single gpu training setup
    ml_class = Model_cons(CONV_LSTM_IMG_INPUT, CONV_LSTM_TM_INPUT, use_norm= NORM_LAYS)
    model = ml_class.gf_CAE_LSTM_tm_model(width=MODEL_WIDTH, l_val= LEAKY_RELU, w_dec=REG_WEIGHT, w_dec_rec=REG_WEIGHT_REC)     #model with soil moisture and time inputs
    # model = ml_class.gf_dia_CAE_LSTM_tm_model(width=MODEL_WIDTH, l_val= LEAKY_RELU, w_dec=REG_WEIGHT)   #with dialated convolution
    # model = ml_class.gf_CAE_LSTM_tm_large_model(width=MODEL_WIDTH, l_val= LEAKY_RELU)     #model with soil moisture and time inputs

################display model summary
model.summary()
plot_model(model, to_file="/home/ubuntu/Oyebade/gap_filling/models/model_archi/gf_Conv_LSTM.pdf", 
    show_shapes=True, show_layer_names=True)

#################split training and testing data
train_data, test_data = split_trainTest_data(X, z, y, test_split= 0.1)
x_train, z_train, y_train = train_data[0], train_data[1], train_data[2]
x_test, z_test, y_test = test_data[0], test_data[1], test_data[2]

#################model callbacks and settings for training
early_stopping = EarlyStopping(monitor="loss", patience=10)
lr_scheduler = LearningRateScheduler(lr_schedule)
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor="loss",    #monitor 'loss' 'val_loss, 'acc' or 'val_acc'
    mode="auto", save_freq= EPOCHS,  save_best_only=True, verbose=0)

if MODEL_CHKPOINT== True:
    call_bcks = [lr_scheduler, early_stopping, model_checkpoint]
else:
    call_bcks = [lr_scheduler, early_stopping]

if MODEL_LOSS == 'mse':
    loss_fn = keras.losses.MeanSquaredError() 
    y_label_plt = 'mean squared error'
elif MODEL_LOSS == 'mae':
    loss_fn = tf.keras.losses.MeanAbsoluteError()   
    y_label_plt = 'mean absolute error'

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule(0))
model.compile(optimizer=optimizer, loss=loss_fn)

history_callback = model.fit([x_train, z_train], y_train, batch_size=BATCH_SZ, epochs=EPOCHS, validation_split=0.2,    
                validation_batch_size=BATCH_SZ_TEST, validation_freq = VAL_FREQ, max_queue_size= 10,
                callbacks=call_bcks, shuffle=True, workers=8, use_multiprocessing=False)

##############save trained model 
if SAVE_TRAINED_MODEL==True:
    model.save(model_name)

##############model evaluation on trained model
test_perf = model.evaluate([x_test, z_test], y_test, batch_size=BATCH_SZ_TEST, workers=8, use_multiprocessing=False)

#############model testing performance evaluation
print('Testing loss:', test_perf)

##############retrieve model training metrics
train_loss = history_callback.history["loss"]
val_loss = history_callback.history["val_loss"]

#############plot and save ml model training evolution
plot_train_hist(train_loss, val_loss, EPOCHS, plot_title='model training', y_label= y_label_plt, 
    x_label='Epochs', name_for_save= trained_model_train_curve, 
    sv_path= save_model_loss_path)

