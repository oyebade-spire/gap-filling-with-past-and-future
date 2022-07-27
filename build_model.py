import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2



class Model_cons(object):
    """ function: returns the ML model for training SMAP gap-filling """
    def __init__(self, img_inp, tm_inp, use_norm=None):
        """ img_inp: ML model input dimension 
        use_norm: flag that determines whether to use layer normalization or not  """
        self.img_inp = img_inp
        self.tm_inp = tm_inp
        self.use_norm = use_norm



    def gf_CAE_LSTM_tm_large_model(self, width=None, l_val= 0.182):
        img_inp = layers.Input(shape=self.img_inp)
        tm_inp = layers.Input(shape=self.tm_inp)
        if width ==1 or width==None:    #default to a width of 1, if value is not provided
            k=1
        else:
            k=width

        # tm_neg_exp = layers.Lambda(lambda x: tf.math.exp(x))(tm_inp * -1.5)
        # x_1 = layers.Multiply()([img_inp, tm_neg_exp])
        ######soil moisture model encoder starts here
        x_1 = layers.ConvLSTM2D(filters=32*k, kernel_size= 10, strides=1, padding='same', activation='relu', return_sequences=True,
        return_state=False, stateful=False)(img_inp)
        # x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
        if self.use_norm==True:
            x_1 = layers.LayerNormalization(axis=-1) (x_1)
        # x_1 = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
        # return_state=False, stateful=False)(x_1)
        # # x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
        # if self.use_norm==True:
        #     x_1 = layers.LayerNormalization(axis=-1) (x_1)
        x_1 = layers.ConvLSTM2D(filters=16*k, kernel_size= 5, strides=1, padding='same', activation='relu', return_sequences=True,
        return_state=False, stateful=False)(x_1)
        # x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
        if self.use_norm==True:
            x_1 = layers.LayerNormalization(axis=-1) (x_1)

        ######time model encoder starts here
        # x_2 = layers.Lambda(lambda x: tf.math.exp(x))(tm_inp * -1.5)
        x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
        return_state=False, stateful=False)(tm_inp)
        # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
        if self.use_norm==True:
            x_2 = layers.LayerNormalization(axis=-1) (x_2)
        # x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        # return_state=False, stateful=False)(x_2)
        # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
        # if self.use_norm==True:
        #     x_2 = layers.LayerNormalization(axis=-1) (x_2)
        # x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        # return_state=False, stateful=False)(x_2)
        # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
        # if self.use_norm==True:
        #     x_2 = layers.LayerNormalization(axis=-1) (x_2)

        #######gating mechanism 1
        C = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
        return_state=False, stateful=False, bias_initializer='zeros')(x_2)
        C = layers.Activation(activation='sigmoid')(C)
        C_g = layers.Lambda(lambda x: 1.0 - x)(C)
        sm_g = layers.Multiply()([x_1, C_g])
        tm_g = layers.Multiply()([x_2, C])

        #######gating mechanism 2
        # C = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        # return_state=False, stateful=False, use_bias=True, activation='relu', bias_initializer=tf.constant_initializer(0.95))(x_1)
        # C = layers.Activation(activation='sigmoid')(C)
        # C_g = layers.Lambda(lambda x: 1.0 - x)(C)
        # sm_g = layers.Multiply()([x_1, C])
        # tm_g = layers.Multiply()([x_2, C_g])
        # x_2 = layers.Lambda(lambda x: tf.math.exp(x, name='exponential weighting'))(x_2 * -1.5)

        #####joint embedding here by concatenation of latent representations
        x = layers.concatenate([sm_g, tm_g], axis=-1)
        # x = layers.concatenate([x_1, x_2], axis=-1)

        ####model decoder starts here
        x = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
        return_state=False, stateful=False)(x)
        # x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        # x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
        # return_state=False, stateful=False)(x)
        # # x = layers.LeakyReLU(alpha=l_val)(x)
        # if self.use_norm==True:
        #     x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=False,
        return_state=False, stateful=False)(x)  #here removes the time-step dimension, so that output is compatible with next layer
        # x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.Conv2D(filters=1, kernel_size=3, strides=(1, 1), padding="same", activation='sigmoid')(x)
        
        model = keras.Model([img_inp, tm_inp], x, name="GF_CAE_LSTM_model")
        return model
    




    def gf_dia_CAE_LSTM_tm_model(self, width=None, l_val= 0.182, w_dec=None):
            img_inp = layers.Input(shape=self.img_inp)
            tm_inp = layers.Input(shape=self.tm_inp)
            if width ==1 or width==None:    #default to a width of 1, if value is not provided
                k=1
            else:
                k=width

            # tm_neg_exp = layers.Lambda(lambda x: tf.math.exp(x))(tm_inp * -1.5)
            # x_1 = layers.Multiply()([img_inp, tm_neg_exp])
            ######soil moisture model encoder starts here
            x_1 = layers.ConvLSTM2D(filters=32*k, kernel_size= 5, strides=1, padding='same', return_sequences=True,
            return_state=False, stateful=False, dilation_rate=2, recurrent_regularizer = l2(w_dec))(img_inp)
            x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
            if self.use_norm==True:
                x_1 = layers.LayerNormalization(axis=-1) (x_1)
            # x_1 = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            # return_state=False, stateful=False, recurrent_regularizer = l2(w_dec))(x_1)
            # x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
            # if self.use_norm==True:
            #     x_1 = layers.LayerNormalization(axis=-1) (x_1)
            x_1 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            return_state=False, stateful=False, dilation_rate=2, recurrent_regularizer = l2(w_dec))(x_1)
            x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
            if self.use_norm==True:
                x_1 = layers.LayerNormalization(axis=-1) (x_1)

            ######time model encoder starts here
            # x_2 = layers.Lambda(lambda x: tf.math.exp(x))(tm_inp * -1.5)
            x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            return_state=False, stateful=False, recurrent_regularizer = l2(w_dec))(tm_inp)
            x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
            if self.use_norm==True:
                x_2 = layers.LayerNormalization(axis=-1) (x_2)
            # x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
            # return_state=False, stateful=False)(x_2)
            # # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
            # if self.use_norm==True:
            #     x_2 = layers.LayerNormalization(axis=-1) (x_2)
            # x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            # return_state=False, stateful=False)(x_2)
            # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
            # if self.use_norm==True:
            #     x_2 = layers.LayerNormalization(axis=-1) (x_2)

            #######gating mechanism 1
            C = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
                return_state=False, stateful=False, bias_initializer='zeros', dilation_rate=2, 
                recurrent_regularizer = l2(w_dec))(x_2)
            C = layers.Activation(activation='sigmoid')(C)
            C_g = layers.Lambda(lambda x: 1.0 - x)(C)
            sm_g = layers.Multiply()([x_1, C_g])
            tm_g = layers.Multiply()([x_2, C])

            #######gating mechanism 2
            # C = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            # return_state=False, stateful=False, use_bias=True, activation='relu', bias_initializer=tf.constant_initializer(0.95))(x_1)
            # C = layers.Activation(activation='sigmoid')(C)
            # C_g = layers.Lambda(lambda x: 1.0 - x)(C)
            # sm_g = layers.Multiply()([x_1, C])
            # tm_g = layers.Multiply()([x_2, C_g])
            # x_2 = layers.Lambda(lambda x: tf.math.exp(x, name='exponential weighting'))(x_2 * -1.5)

            #####joint embedding here by concatenation of latent representations
            x = layers.concatenate([sm_g, tm_g], axis=-1)
            # x = layers.concatenate([x_1, x_2], axis=-1)

            ####model decoder starts here
            x = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            return_state=False, stateful=False, dilation_rate=2, recurrent_regularizer = l2(w_dec))(x)
            x = layers.LeakyReLU(alpha=l_val)(x)
            if self.use_norm==True:
                x = layers.LayerNormalization(axis=-1) (x)
            # x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
            # return_state=False, stateful=False, recurrent_regularizer = l2(w_dec))(x)
            # x = layers.LeakyReLU(alpha=l_val)(x)
            # if self.use_norm==True:
            #     x = layers.LayerNormalization(axis=-1) (x)
            x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=False,
            return_state=False, stateful=False, dilation_rate=2, recurrent_regularizer = l2(w_dec))(x)  #here removes the time-step dimension, so that output is compatible with next layer
            x = layers.LeakyReLU(alpha=l_val)(x)
            if self.use_norm==True:
                x = layers.LayerNormalization(axis=-1) (x)
            x = layers.Conv2D(filters=1, kernel_size=3, dilation_rate=2, strides=(1, 1), padding="same", activation='sigmoid')(x)
            
            model = keras.Model([img_inp, tm_inp], x, name="GF_CAE_LSTM_model")
            return model




    def gf_CAE_LSTM_tm_model(self, width=None, l_val= 0.182, w_dec=None, w_dec_rec=None):
        img_inp = layers.Input(shape=self.img_inp)
        tm_inp = layers.Input(shape=self.tm_inp)
        if width ==1 or width==None:    #default to a width of 1, if value is not provided
            k=1
        else:
            k=width

        # tm_neg_exp = layers.Lambda(lambda x: tf.math.exp(x))(tm_inp * -1.5)
        # x_1 = layers.Multiply()([img_inp, tm_neg_exp])
        ######soil moisture model encoder starts here
        x_1 = layers.ConvLSTM2D(filters=64*k, kernel_size= 5, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(img_inp)
        x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
        if self.use_norm==True:
            x_1 = layers.LayerNormalization(axis=-1) (x_1)
        x_1 = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(x_1)
        x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
        if self.use_norm==True:
            x_1 = layers.LayerNormalization(axis=-1) (x_1)
        x_1 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(x_1)
        x_1 = layers.LeakyReLU(alpha=l_val)(x_1)
        if self.use_norm==True:
            x_1 = layers.LayerNormalization(axis=-1) (x_1)

        ######time model encoder starts here
        # x_2 = layers.Lambda(lambda x: tf.math.exp(x))(tm_inp * -1.5)
        x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(tm_inp)
        x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
        if self.use_norm==True:
            x_2 = layers.LayerNormalization(axis=-1) (x_2)
        # x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', activation='relu', return_sequences=True,
        # return_state=False, stateful=False)(x_2)
        # # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
        # if self.use_norm==True:
        #     x_2 = layers.LayerNormalization(axis=-1) (x_2)
        # x_2 = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        # return_state=False, stateful=False)(x_2)
        # x_2 = layers.LeakyReLU(alpha=l_val)(x_2)
        # if self.use_norm==True:
        #     x_2 = layers.LayerNormalization(axis=-1) (x_2)

        #######gating mechanism 1
        C = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, bias_initializer='zeros', kernel_regularizer=l2(w_dec),
            recurrent_regularizer = l2(w_dec_rec))(x_2)
        C = layers.Activation(activation='sigmoid')(C)
        C_g = layers.Lambda(lambda x: 1.0 - x)(C)
        sm_g = layers.Multiply()([x_1, C_g])
        tm_g = layers.Multiply()([x_2, C])

        #######gating mechanism 2
        # C = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        # return_state=False, stateful=False, use_bias=True, activation='relu', bias_initializer=tf.constant_initializer(0.95))(x_1)
        # C = layers.Activation(activation='sigmoid')(C)
        # C_g = layers.Lambda(lambda x: 1.0 - x)(C)
        # sm_g = layers.Multiply()([x_1, C])
        # tm_g = layers.Multiply()([x_2, C_g])
        # x_2 = layers.Lambda(lambda x: tf.math.exp(x, name='exponential weighting'))(x_2 * -1.5)

        #####joint embedding here by concatenation of latent representations
        x = layers.concatenate([sm_g, tm_g], axis=-1)
        # x = layers.concatenate([x_1, x_2], axis=-1)

        ####model decoder starts here
        x = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(x)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(x)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=64*k, kernel_size= 3, strides=1, padding='same', return_sequences=False,
        return_state=False, stateful=False, kernel_regularizer=l2(w_dec), recurrent_regularizer = l2(w_dec_rec))(x)  #here removes the time-step dimension, so that output is compatible with next layer
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding="same", kernel_regularizer=l2(w_dec_rec), activation='sigmoid')(x)
        
        model = keras.Model([img_inp, tm_inp], x, name="GF_CAE_LSTM_model")
        return model





    def gf_CAE_LSTM_model(self, width=None, l_val= 0.182):
        img_inp = layers.Input(shape=self.img_inp)
        if width ==1 or width==None:    #default to a width of 1, if value is not provided
            k=1
        else:
            k=width
        x = layers.ConvLSTM2D(filters=64*k, kernel_size= 5, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False)(img_inp)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False)(x)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False)(x)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=16*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False)(x)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=32*k, kernel_size= 3, strides=1, padding='same', return_sequences=True,
        return_state=False, stateful=False)(x)
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.ConvLSTM2D(filters=64*k, kernel_size= 3, strides=1, padding='same', return_sequences=False,
        return_state=False, stateful=False)(x)  #here removes the time-step dimension, so that output is compatible with next layer
        x = layers.LeakyReLU(alpha=l_val)(x)
        if self.use_norm==True:
            x = layers.LayerNormalization(axis=-1) (x)
        x = layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding="same", activation='sigmoid')(x)
        
        model = keras.Model(img_inp, x, name="GF_CAE_LSTM_model")
        return model




    