import tensorflow as tf

import os
import math
import numpy as np
import numpy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping,ModelCheckpoint
import tensorflow_datasets as tfds
from build_model import model_buildCNN
from data_utilities import DataSequence_train, DataSequence_test, get_filenames, create_sequences
from train_utilities import lr_schedule
from data_utilities_tf import preprocessing, create_sequences

# for allowing the growth of gpu resources
gpus = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpus[0], True)



DATA_PIPELINE = 'keras_seq'     #options are 'keras_seq' or 'tf.data'
NORM_SCALE = 1.0                #if set to value other than 1.0, it will be used for normalizing the image
EPOCHS = 50    #training epochs
BATCH_SZ = 2     #training batch size
BATCH_SZ_TEST = 4    #testing batch size
IMG_SZ = 20        #input image size
UPSAMP_FAC = 1       #must be a factor of 2
UPSAMP_METHOD = 'trans_conv'  #UPSAMP_METHOD='trans_conv' or 'shuff_pix'
VAL_FREQ = 2
DEPTH              = 1  #model depth
WIDE               = 4  #model width; must be greater than 2
IMG_ROWS, IMG_COLS = None, None        #use the relation or set as None
IMG_CHANNELS       = 1  #number of input channels
OUT_CHANNELS       = 1  #number of output channels
W_DECAY = 0      #weight decay
IMG_INPUT = (IMG_ROWS, IMG_COLS, IMG_CHANNELS)  #input image shape
TRANS_FAC = 1     #trans_fac: factor for scaling the learned transformation before addition to the shortcut


###########
g = model_buildCNN(DEPTH, WIDE, UPSAMP_FAC, TRANS_FAC, W_DECAY)
SR_CAE = g.ae_SR_model(IMG_INPUT, OUT_CHANNELS, UPSAMP_METHOD)
SR_CAE.summary()

""" #lists all the layers inside SR_CAE
for layer in SR_CAE.layers:
    print(layer.output_shape) """

""" #lists all the layers in the decoder model inside SR_CAE
for layer in SR_CAE.layers[-1].layers:
    print(layer.output_shape) """


######get filenames for training, validation and testing data
training_images = []   
training_img_path = 'C:\\Users\\Oyebade Oyedotun\\Documents\\Spire_soil_moisture_codes\\gap_filling\\smap_tiff\\train_inp'
val_img_path = 'C:\\Users\\Oyebade Oyedotun\\Documents\\Spire_soil_moisture_codes\\gap_filling\\smap_tiff\\train_out'
testing_img_path = 'C:\\Users\\Oyebade Oyedotun\\Documents\\Spire_soil_moisture_codes\\gap_filling\\smap_tiff\\train_out'

training_images = get_filenames(training_img_path, sort_ls=False)
val_images = get_filenames(val_img_path, sort_ls=True)
testing_images = get_filenames(testing_img_path, sort_ls=True)


#####training/testing data break down    
print('number of training samples:', len(training_images))
print('number of validation labels:', len(val_images))  
print('number of testing samples:', len(testing_images))

perm = len(training_images)
perm = np.random.permutation(perm)     #permutate sample indices
perm = list(perm)       #convert to list

training_images = [training_images[index] for index in perm]

print("Data is ready...") 

if DATA_PIPELINE== 'keras_seq':
    # training data
    train_gen = DataSequence_train(training_images, training_images, BATCH_SZ, shuffle=True)
    # Validation data
    val_gen = DataSequence_test(val_images, val_images, BATCH_SZ_TEST)
    # training data
    testing_gen = DataSequence_test(testing_images, testing_images, BATCH_SZ_TEST)
elif DATA_PIPELINE== 'tf.data':
    train_gen = preprocessing(training_images, batch_size=BATCH_SZ, repeat_count=None, training=True)
    val_gen = preprocessing(val_images, batch_size=BATCH_SZ_TEST, repeat_count=None, training=False)
    testing_gen = preprocessing(testing_images, batch_size=BATCH_SZ_TEST, repeat_count=None, training=False)


x_train = create_sequences(train_gen)
print("Training input shape: ", x_train.shape)  


# callbacks for training
early_stopping = EarlyStopping(monitor="loss", patience=10)
lr_scheduler = LearningRateScheduler(lr_schedule)

# checkpoint_filepath = "./tmp/checkpoint"
# checkpoint_filepath = "./checkpoint/SR_CAE_checkpoint-{epoch:02d}-{loss:.4f}.hdf5"
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint_filepath="./checkpoint/GF_CAE_checkpoint.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor="loss",    #monitor 'loss' 'val_loss, 'acc' or 'val_acc'
    mode="auto", save_freq= EPOCHS,  save_best_only=True, verbose=0)

call_bcks = [lr_scheduler, early_stopping, model_checkpoint]

# loss_fn = keras.losses.MeanSquaredError()
loss_fn = tf.keras.losses.MeanAbsoluteError()


optimizer = keras.optimizers.Adam(learning_rate=lr_schedule(0))
SR_CAE.compile(optimizer=optimizer, loss=loss_fn)

STEP_SIZE_TRAIN = len(training_images)//BATCH_SZ
STEP_SIZE_VALID = len(val_images)//BATCH_SZ_TEST
STEP_SIZE_TEST = len(testing_images)//BATCH_SZ_TEST
history_callback = SR_CAE.fit(train_gen, 
                steps_per_epoch=STEP_SIZE_TRAIN, epochs=EPOCHS, 
                validation_data=val_gen,    #trick training so that we can checkpoint model
                validation_steps=STEP_SIZE_VALID, #trick training so that we can checkpoint model
                validation_freq = VAL_FREQ,
                callbacks=call_bcks, shuffle=True, workers=8, use_multiprocessing=False)


# save trained model 
model_name = './checkpoint/GF_CAE_end.h5'
SR_CAE.save(model_name)

# model evaluation on trained model
test_perf = SR_CAE.evaluate(testing_gen, steps=STEP_SIZE_TEST, workers=8, use_multiprocessing=False)




# model testing performance evaluation
print('Testing loss:', test_perf)