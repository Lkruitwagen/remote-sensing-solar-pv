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

## but what if we tried this for 372x372?

def DEEP_RESUNET(input_shape):
    
    inputs = Input(input_shape)

    ###encoding block 1
    iden1 = Conv2D(32, 1, activation = None, padding='same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(inputs) #200
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1) #200
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1) #200
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    add1 = Add()([iden1,conv1])
    pool1 = MaxPooling2D()(add1) #200 -> 100 x 32
    print(add1)
    
    ###encoding block2

    conv2 = BatchNormalization()(pool1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2) #200
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2) #200
    add2 = Add()([pool1,conv2])
    pool2 = MaxPooling2D()(add2) #100 ->50 x 32
    print (add2)
    
    ###encoding block3
    conv3 = BatchNormalization()(pool2)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3) #200
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3) #200
    add3 = Add()([pool2,conv3])
    pool3= MaxPooling2D()(add3) #50->25 x 32
    print (add3)
    
    ###encoding block4
    conv4 = BatchNormalization()(pool3)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4) #200
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv4) #200
    add4 = Add()([pool3,conv4])
    pool4 = MaxPooling2D()(add4) #25->12
    print (pool4)
    
    ###encoding block5
    conv5 = BatchNormalization()(pool4)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #200
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    conv5 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5) #200
    add5 = Add()([pool4, conv5])
    pool5 = MaxPooling2D()(add5) #12->6
    print (pool5)
    
    ###encoding block6
    conv6 = BatchNormalization()(pool5)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6) #200
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6) #200
    add6 = Add()([pool5, conv6])
    drop6 = Dropout(0.5)(add6)
    pool6 = MaxPooling2D()(drop6) #6->3
    print (pool6)
    
    ###bridge

    conv7 = BatchNormalization()(pool6)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7) #200
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    conv7 = Conv2D(32, 2, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv7) #200
    drop7 = Dropout(0.5)(conv7)
    print (drop7)
    
    ###decoding block1
    up8 = UpSampling2D()(drop7) #3->6 x 32
    #up8 = ZeroPadding2D(((1,0),(1,0)))(up6) #24->25
    concat8 = Concatenate(axis=3)([up8,add6]) #6x64
    conv8 = BatchNormalization()(concat8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8) #200
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    conv8 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv8) #200
    add8 = Add()([up8,conv8])
    
    ###decoding block2
    up9 = UpSampling2D()(add8) #6->12 x 32
    #up8 = ZeroPadding2D(((1,0),(1,0)))(up6) #24->25
    concat9 = Concatenate(axis=3)([up9,add5]) #12x64
    conv9 = BatchNormalization()(concat9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9) #200
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)
    conv9 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv9) #200
    add9 = Add()([up9,conv9])
    
    ###decoding block3
    up10 = UpSampling2D()(add9) #12->24 x 32
    up10 = ZeroPadding2D(((1,0),(1,0)))(up10) #24->25
    concat10 = Concatenate(axis=3)([up10,add4]) #25x64
    conv10 = BatchNormalization()(concat10)
    conv10 = Activation('relu')(conv10)
    conv10 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv10) #200
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)
    conv10 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv10) #200
    add10 = Add()([up10,conv10])
    
     ###decoding block4
    up11 = UpSampling2D()(add10) #25->50
    concat11 = Concatenate(axis=3)([up11,add3]) #50x64
    conv11 = BatchNormalization()(concat11)
    conv11 = Activation('relu')(conv11)
    conv11 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv11) #200
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)
    conv11 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv11) #200
    add11 = Add()([up11,conv11])
    
     ###decoding block5
    up12 = UpSampling2D()(add11) #50->100
    concat12 = Concatenate(axis=3)([up12,add2]) #50x64
    conv12 = BatchNormalization()(concat12)
    conv12 = Activation('relu')(conv12)
    conv12 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv12) #200
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)
    conv12 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv12) #200
    add12 = Add()([up12,conv12])
    
     ###decoding block6
    up13 = UpSampling2D()(add12) #100->200
    concat13 = Concatenate(axis=3)([up13,add1]) #200x64
    conv13 = BatchNormalization()(concat13)
    conv13 = Activation('relu')(conv13)
    conv13 = Conv2D(64, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv13) #200
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)
    conv13 = Conv2D(32, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv13) #200
    add13 = Add()([up13,conv13])
    
    
    
    
    conv14 = Conv2D(N_CLASSES, 3, activation ='sigmoid', padding = 'same', kernel_initializer = 'he_normal')(add13)
    #sigmoid probably too strong an activation
    
    model = Model(input = inputs, output = conv14)

    return model
    
    
    
deep_resunet_model = DEEP_RESUNET((200,200,14))
print (deep_resunet_model.summary())


with open('model_deep_resunet.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(deep_resunet_model.to_json()), indent=2))
