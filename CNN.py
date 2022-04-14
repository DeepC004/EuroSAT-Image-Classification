import cv2
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout 
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from matplotlib import pyplot as plt
import splitfolders
import pandas as pd
from keras.utils import np_utils
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf_devices='/gpu:0'

X_test = []
Y_test = []
testPath = 'E:\CSLearning\AI and ML\Projects\Satellite Image Classification\Pictures\Output\\test'
for folder in os.listdir(testPath):
    for image in os.listdir(os.path.join(testPath, folder)):
        X_test.append(cv2.imread(os.path.join(testPath, os.path.join(folder,image))))
        Y_test.append(folder)

X_train = []
Y_train = []
trainPath = 'E:\CSLearning\AI and ML\Projects\Satellite Image Classification\Pictures\Output\\train'
for folder in os.listdir(trainPath):
    for image in os.listdir(os.path.join(trainPath, folder)):
        X_train.append(cv2.imread(os.path.join(trainPath, os.path.join(folder,image))))
        Y_train.append(folder)

Y_train = np.asarray(Y_train)
X_test = np.asarray(X_test)
X_train = np.asarray(X_train)
Y_test = np.asarray(Y_test)

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(64,64,3), padding='same', activation='relu'))
model.add(Dropout(0.15))
model.add(Conv2D(32, (3,3), input_shape=(64,64,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2,strides=2,padding='valid'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.summary()

import random 
buf = list(zip(X_train,Y_train))
random.shuffle(buf)
X_train,Y_train = zip(*buf)
buf = list(zip(X_test,Y_test))
random.shuffle(buf)
X_test,Y_test = zip(*buf)

Y_train=np.asarray(Y_train)
X_test=np.asarray(X_test)
X_train = np.asarray(X_train)
Y_test=np.asarray(Y_test)
print(X_train.shape)
print(X_test.shape)

results = {
    'AnnualCrop':0,
    'Forest':1,
    'HerbaceousVegetation':2,
    'Highway':3,
    'Industrial':4,
    'Pasture':5,
    'PermanentCrop':6,
    'Residential':7,
    'River':8,
    'SeaLake':9
}
y_train = []
for i in Y_train:
    y_train.append(results.get(i))

y_train = np_utils.to_categorical(y_train)
print(y_train[0])

scale = X_train.max()
X_train = X_train/scale
X_test = X_test/scale

mean = X_train.mean()
X_train = X_train - mean
X_test = X_test - mean

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1,verbose=1)