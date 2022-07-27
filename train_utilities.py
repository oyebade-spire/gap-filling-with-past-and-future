import tensorflow as tf
from tensorflow import keras
import numpy as np


'''
learning schedule
'''        
def lr_schedule(epoch):
        """Learning Rate Schedule
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
        # Arguments
            epoch (int): The number of epochs
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 8:
            lr *= 1e-4
        elif epoch > 6:
            lr *= 1e-3
        elif epoch > 4:
            lr *= 1e-2
        elif epoch > 2:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr 



# class ESPCNCallback(keras.callbacks.Callback):
#     def __init__(self):
#         super(ESPCNCallback, self).__init__()
#         self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

#     # Store PSNR value in each epoch.
#     def on_epoch_begin(self, epoch, logs=None):
#         self.psnr = []

#     def on_epoch_end(self, epoch, logs=None):
#         print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
#         if epoch % 20 == 0:
#             prediction = upscale_image(self.model, self.test_img)
#             plot_results(prediction, "epoch-" + str(epoch), "prediction")

#     def on_test_batch_end(self, batch, logs=None):
#         self.psnr.append(10 * math.log10(1 / logs["loss"]))