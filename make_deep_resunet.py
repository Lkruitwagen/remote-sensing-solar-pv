import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, Cropping2D, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Concatenate
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, UpSampling2D, Add
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

def RESUNET(input_shape):
    
    inputs = Input(input_shape)


    ###encoding block 1
    iden1 = Conv2D(64, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs) #200
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1) #200
    add1 = Add()([iden1,conv1])
    pool1 = MaxPooling2D()(add1) #200 -> 100
    print(add1)
    
    ###encoding block2
    iden2 = Conv2D(128, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2) #200
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2) #200
    add2 = Add()([iden2,conv2])
    pool2 = MaxPooling2D()(add2) #100 ->50
    print (add2)
    
    ###encoding block3
    iden3 = Conv2D(256, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3) #200
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3) #200
    add3 = Add()([iden3,conv3])
    pool3= MaxPooling2D()(add3) #50->25
    print (add3)
    
    ###encoding block4
    iden4 = Conv2D(512, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4) #200
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4) #200
    add4 = Add()([iden4,conv4])
    drop4 = Dropout(0.5)(add4)
    pool4 = MaxPooling2D()(drop4) #25->12
    print (pool4)
    
    ###bridge
    conv5 = BatchNormalization()(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1014, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #200
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #200
    drop5 = Dropout(0.5)(conv5)
    print (conv5)
    
    ###decoding block1
    up6 = UpSampling2D()(drop5) #12->24
    up6 = ZeroPadding2D(((1,0),(1,0)))(up6) #24->25
    concat6 = Concatenate(axis=3)([up6,add4])
    iden6 = Conv2D(512, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(concat6)
    conv6 = BatchNormalization()(concat6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6) #200
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6) #200
    add6 = Add()([iden6,conv6])
    
    ###decoding block2
    up7 = UpSampling2D()(add6) #25->50
    #up7 = ZeroPadding2D(((1,0),(1,0)))(up7) 24->25
    concat7 = Concatenate(axis=3)([up7,add3])
    iden7 = Conv2D(256, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(concat7)
    conv7 = BatchNormalization()(concat7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7) #200
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7) #200
    add7 = Add()([iden7,conv7])
    
    ###decoding block3
    up8 = UpSampling2D()(add7) #50->100

    concat8 = Concatenate(axis=3)([up8,add2])
    iden8 = Conv2D(128, 1, activation=None, padding='same', kernel_initializer = 'he_normal')(concat8)
    conv8 = BatchNormalization()(concat8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8) #200
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8) #200
    add8 = Add()([iden8,conv8])
    
    ###decoding block4
    up9 = UpSampling2D()(add8) #100->200
    #up7 = ZeroPadding2D(((1,0),(1,0)))(up7) 24->25
    concat9 = Concatenate(axis=3)([up9,add1])
    iden9 = Conv2D(64,1,activation=None, padding='same', kernel_initializer = 'he_normal')(concat9)
    conv9 = BatchNormalization()(concat9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9) #200
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9) #200
    add9 = Add()([iden9,conv9])
    
    conv10 = Conv2D(N_CLASSES, 3, activation ='sigmoid', padding = 'same', kernel_initializer = 'he_normal')(add9)
    #sigmoid probably too strong an activation
    
    model = Model(input = inputs, output = conv10)

    return model
    
    
    
resunet_model = RESUNET((200,200,14))
print (resunet_model.summary())


with open('model_resunet.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(resunet_model.to_json()), indent=2))
