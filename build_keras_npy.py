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

person_catego = 'YoungHwan SoBin'.split()
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
    if g == 'YoungHwan':
        continue
    if g == 'SoBin':
        bound = 5
    for filename in os.listdir(f'./img_data/{g}'):
        if filename == '.DS_Store':
            continue
        
        Username = f'./img_data/{g}/{filename}'
        img = load_img(Username)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
 
        i = 0
        for batch in data_datagen.flow(x,save_to_dir=f'./img_data/{g}', save_prefix=g, save_format='jpg'):
            i += 1
            if i > bound:
                break
        print('file dunp complete'+ filename)
print('finish')

X = [] 
Y = [] 

for idx,g in enumerate(person_catego):
    for filename in os.listdir(f'./img_data/{g}'):
        if filename == '.DS_Store':
            continue
        name = f'./img_data/{g}/{filename}'
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
np.save("person_catego.npy", xy)