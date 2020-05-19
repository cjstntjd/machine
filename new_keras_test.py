from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
import os, glob
import numpy as np
import sys, os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import keras
image_size = 64

person_catego = 'YoungHwan SoBin'.split()
nb_classes = len(person_catego)

image_width =64
image_height =64

def load_dataset():
    x_train, x_test, y_train, y_test = np.load("person_catego.npy",allow_pickle=True)
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return  x_train, x_test, y_train, y_test

def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='Same',kernel_initializer='he_uniform',
                input_shape=in_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                    optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9), 
                    metrics=['accuracy'])

    return model


def model_train(x, y):
    model = build_model(x.shape[1:])
    model.fit(x, y, batch_size=32, epochs=30)

    return model

# 모델 평가하기  (4)
def model_eval(model, x, y):
    score = model.evaluate(x, y)
    print('loss=', score[0])
    print('accuracy=', score[1])

x_train, x_test, y_train, y_test = load_dataset()
model = model_train(x_train, y_train)
model_eval(model, x_test, y_test)

model.save("UL_Face.h5")