import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def unet(pretrained_weights = None, input_size = (284, 284, 3)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size = (2, 2))(conv1)
          
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size = (2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size = (2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size = (2, 2))(conv4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
    
    up6 = Conv2DTranspose(512, 2, strides = 2, padding = 'valid', kernel_initializer = 'he_normal')(conv5)     
    s = round((keras.int_shape(conv4)[1] - keras.int_shape(up6)[1])/2)
    crop6 = Cropping2D(cropping = s)(conv4)
    up6 = concatenate([crop6,up6], axis = -1)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(up6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
    
    up7 = Conv2DTranspose(256, 2, strides = 2, padding = 'valid', kernel_initializer = 'he_normal')(conv6)
    s = round((keras.int_shape(conv3)[1] - keras.int_shape(up7)[1])/2)
    crop7 = Cropping2D(cropping = s)(conv3)
    up7 = concatenate([crop7,up7], axis = -1)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(up7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
    
    up8 = Conv2DTranspose(128, 2, strides = 2, padding = 'valid', kernel_initializer = 'he_normal')(conv7)
    s = round((keras.int_shape(conv2)[1] - keras.int_shape(up8)[1])/2)
    crop8 = Cropping2D(cropping = s)(conv2)
    up8 = concatenate([crop8,up8], axis = -1)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(up8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2DTranspose(64, 2, strides = 2, padding = 'valid', kernel_initializer = 'he_normal')(conv8)
    s = round((keras.int_shape(conv1)[1] - keras.int_shape(up9)[1])/2)
    crop9 = Cropping2D(cropping = s)(conv1)
    up9 = concatenate([crop9,up9], axis = -1)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(up9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
    
    conv10 = Conv2D(3, 1, padding = 'valid')(conv9)
   

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_error', metrics = ['accuracy'])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model