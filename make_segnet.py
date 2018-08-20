##https://github.com/imlab-uiip/keras-segnet/blob/master/build_model.py

from keras import models
from keras.layers import ZeroPadding2D
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
import json

img_w = 200
img_h = 200
n_labels = 2

kernel = 3

encoding_layers = [
    Conv2D(64, kernel, padding='same', input_shape=( img_h, img_w,14)),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(128, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(256, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),

    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
]

autoencoder = models.Sequential()
autoencoder.encoding_layers = encoding_layers

for l in autoencoder.encoding_layers:
    autoencoder.add(l)
    print('enc',l.input_shape,l.output_shape,l)
    
print ('onwards')

decoding_layers = [
    UpSampling2D(),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(512, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    ZeroPadding2D(padding=(1,1)),
    Conv2D(256, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(256, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(128, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(128, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(64, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),

    UpSampling2D(),
    Conv2D(64, kernel, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    Conv2D(n_labels, 1, padding='valid'),
    BatchNormalization(),
]
autoencoder.decoding_layers = decoding_layers
for l in autoencoder.decoding_layers:
    autoencoder.add(l)
    print (l.input_shape, l.output_shape, l)
    
    

#autoencoder.add(Reshape((n_labels, img_h * img_w)))
#autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('sigmoid'))

print (autoencoder.summary())

with open('model_segnet.json', 'w') as outfile:
    outfile.write(json.dumps(json.loads(autoencoder.to_json()), indent=2))