import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Cropping2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
#from FCN_utils import *
from keras.metrics import categorical_accuracy

#from sklearn.metrics import confusion_matrix

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt

import json

N_CLASSES = 2

def UNET(input_shape):
    
    inputs = Input(input_shape)


    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) #200

    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1) #200

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #200 -> 100


    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1) #98

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) #96

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #100 -> 50


    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) #48

    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) #48

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #25


    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) #24
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) #24
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) #12

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4) #12
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) #12
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5)) #24
    up6 = ZeroPadding2D(padding=((1,0),(1,0)))(up6)
    merge6 = Concatenate(axis=3)([drop4,up6]) #25,25
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6) #25
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) #25
    print ('cony6 shape', conv6)
    
    up7 = UpSampling2D(size=(2,2))(conv6) #50
    print ('up7 shape', up7)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(up7) #50
    merge7 = Concatenate(axis=3)([conv3,up7]) #50,50
    #conv7 = ZeroPadding2D(padding=(1,1))(merge7)
    #print ('cony7 shape', conv7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7) #50
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) #50

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis=3)([conv2,up8]) #100,100
    print ('merge_8', merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8) #100
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8) #100

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) #200
    merge9 = Concatenate(axis=3)([conv1,up9]) #200
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(N_CLASSES, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #                                          sigmoid
    #try softmax to match segnet?
    conv9 = Conv2D(N_CLASSES, 3, activation ='sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #sigmoid probably too strong an activation
    
    model = Model(input = inputs, output = conv9)

    return model

unet_model = UNET((200,200,14))
print (unet_model.summary())


with open('model_unet.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(unet_model.to_json()), indent=2))
