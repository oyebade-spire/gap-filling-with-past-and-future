import os
import numpy as np
import random, math
from PIL import Image, ImageFilter
import tensorflow as tf
import tensorflow_addons as tfa



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
    image = tf.io.read_file(img)
    image = tf.image.decode_jpeg(image, channels= 1, dct_method='INTEGER_ACCURATE')     #read image as unit8 tensor
    image = tf.image.convert_image_dtype(image, tf.float32) #normalizes the image using the max value in the image
    return image


def process_test(img):
    image = tf.io.read_file(img)
    image = tf.image.decode_jpeg(image, channels= 1, dct_method='INTEGER_ACCURATE')
    image = tf.image.convert_image_dtype(image, tf.float32)     #normalizes the image using the max value in the image
    return image


def preprocessing(data, batch_size=16, repeat_count=None, training=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = tf.data.Dataset.list_files(data)
    # random transform will apply only on train dataset
    if training==True:
        data = data.map(lambda x: process_train(x), num_parallel_calls= AUTOTUNE)
    elif training==False:
        data = data.map(lambda x: process_test(x), num_parallel_calls= AUTOTUNE)
    data = data.batch(batch_size)
    # it will repeat dataset. if repeat_count() = None then infinite cardinality
    data = data.repeat(repeat_count)
    # ds.prefetch() will prefetch data for given buffer_size
    data = data.prefetch(buffer_size=AUTOTUNE)
    
    return data


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=None):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)