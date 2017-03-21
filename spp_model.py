# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:14:37 2017

@author: JunTaniguchi
"""

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import ZeroPadding1D
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from sklearn import cross_validation

def spp_model(input_shape, NUM_CLASSES, optim):
    from SpatialPyramidPooling import SpatialPyramidPooling

    # VGG16（RGB）
    
    model = Sequential()
    model.add(ZeroPadding2D((2,2),input_shape=(None, None, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((2,2)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dropout(0.25))
    
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=['accuracy'])
    model.summary()
    
    '''
    # 単純NN
    model = Sequential()
    model.add(Dense(32,input_shape=(None, None, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(SpatialPyramidPooling([1, 2, 4]))

    model.add(Dense(NUM_CLASSES, activation='relu'))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    model.summary()
        
    '''
    
    return model