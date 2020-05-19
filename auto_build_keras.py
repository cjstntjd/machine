import os, glob
import numpy as np
import sys, os
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from PIL import Image
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

def load_dataset():
    x_train, x_test, y_train, y_test = np.load("person_catego.npy",allow_pickle=True)
    x_train = x_train.astype("float") / 256
    x_test = x_test.astype("float") / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return  x_train, x_test, y_train, y_test

def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='Same', 
                input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', 
                    optimizer='rmsprop', 
                    metrics=['accuracy'])

    return model

def model_train(x, y):
    model = build_model(x.shape[1:])
    model.fit(x, y, batch_size=32, epochs=30)

    return model

def model_eval(model, x, y):
    score = model.evaluate(x, y)
    print('loss=', score[0])
    print('accuracy=', score[1])


person_catego = ''

for foldername in os.listdir(f'./UserData'):
    if foldername =='.DS_Store':
        continue
    person_catego+=foldername
    person_catego+=' '
person_catego = person_catego.split()
print(person_catego)
nb_classes = len(person_catego)

image_width =64
image_height =64

data_datagen = ImageDataGenerator(rescale=1./255)
 
data_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=30,
                                     
                                   width_shift_range=0.2,
                                  height_shift_range=0.2,
                                   
                                
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest') 
for idx,g in enumerate(person_catego):
    for filename in os.listdir(f'./UserData/{g}'):
        if filename == '.DS_Store':
            continue
        Username = f'./UserData/{g}/{filename}'
        img = load_img(Username)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
 
        i = 0
        for batch in data_datagen.flow(x,save_to_dir=f'./UserData/{g}', save_prefix=g, save_format='jpg'):
            i += 1
            if i > 10:
                break
        print('image generate step fin : '+str(filename))
print('finish')

X = [] 
Y = [] 

for idx,g in enumerate(person_catego):
    for filename in os.listdir(f'./UserData/{g}'):
        if filename == '.DS_Store':
            continue
        name = f'./UserData/{g}/{filename}'
        img = Image.open(name)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)
        X.append(data)
        Y.append(idx)

print(len(Y))

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

xy = (X_train, X_test, Y_train, Y_test)
# 데이터 파일 저장
np.save("UL_G_emc_face.npy", xy)

x_train, x_test, y_train, y_test = load_dataset()
model = model_train(x_train, y_train)
model_eval(model, x_test, y_test)

model.save("UL_g_emc.h5")