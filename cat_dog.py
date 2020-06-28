# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:52:59 2020

@author: DELL
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog as fd
import os
import cv2
import random
# os.chdir()
filepath=fd.askdirectory() # getting the mother file directory
categories=["Dog", "Cat"] # sub directory 
for category in categories:
    path=os.path.join(filepath,category) #joingin both directories
    for img in os.listdir(path):
        img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) # creating path for each image and reading the image and converting it to gryscale image
        break
    break
img_size=100

training_data=[]
def create_training_data():
     for category in categories:
        path=os.path.join(filepath,category) #joingin both directories
        class_num=categories.index(category) #label categories i.e dog=0, cat=1
        for img in os.listdir(path):
            try:
                img_array= cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE) # creating path for each image and reading the image and converting it to gryscale image           
                new_array=cv2.resize(img_array,(img_size, img_size)) # make the size uniform for all data
                training_data.append([new_array,class_num]) # append all data with its label
            except Exception as e:
                pass # error handling 
                
create_training_data() # calling the function to perform the create_training_data operation
random.shuffle(training_data) #suffling all the sample data
X=[] # list of features (in this case its pixel values)
y=[] # list of labels (in this case its 0 for dog and 1 for cat)
for features, labels in training_data:
    X.append(features)
    y.append(labels)

X=np.array(X).reshape(-1,img_size, img_size,1) # converting features into array with size equal to img_size,-1 is no. of images which means it can be any number, 1 means grayscale    
np.save('cat_n_dog_features.npy',X) #save the preprocessed feature dataset before training
np.save('cat_n_dog_Label.npy',y) # #save the preprocessed label dataset before training
#X=np.load('features.npy') # load a saved data
#%%
#======================================================================================
# APPLYIG CNN TO THE PROCESSED DATA
#======================================================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
from tkinter import filedialog as fd
import os

dataset_path=fd.askdirectory() # getting the mother file directory
os.chdir(dataset_path) # changing diractory to the dataset location
X=np.load('cat_n_dog_features.npy') # loading feature data
y=np.load('cat_n_dog_Label.npy') # loading label data

x_norm=X/255 # normalizing dataset (mannually dividing data by the max value)
#X_norm=tf.keras.utils.normalize(X) # normalizing dataset using inbuilt func.

model=Sequential()

#             **************first layer CNN************

model.add(Conv2D(64,(3,3),input_shape=x_norm.shape[1:])) #  64 neurons, 3x3 win_size, shape same as x_norm
model.add(Activation("relu")) # Activation is rectified linear unit
model.add(MaxPooling2D(pool_size=(2,2))) # maxpooling with window size 2x2




#             **************Second layer CNN************

model.add(Conv2D(64,(3,3))) #  64 neurons, 3x3 win_size, no need to specify shape anymore
model.add(Activation("relu")) # Activation is rectified linear unit
model.add(MaxPooling2D(pool_size=(2,2))) # maxpooling with window size 2x2



#             **************Third layer CNN************

model.add(Flatten()) # Flattening the 2D array for the output layer
model.add(Dense(64))
model.add(Activation("relu"))


#             **************output layer CNN************

model.add(Dense(1))
model.add(Activation('sigmoid'))


#             **************Compilation parameter************

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(x_norm, y, epochs=5, batch_size=32, validation_split=0.1) # fitting/suppyling dataset into model,ephocs=no. of itteration batch_size= no. of samples it takes as input in one go.

model.save('cat_n_dog_classifier.model') # saving the model



#           ************* testing model with new data ************
import tensorflow as tf

import cv2
categories=["dog","cat"]
def prepare(filepath):
    img_size=100
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)

model=tf.keras.models.load_model('cat_n_dog_classifier.model')
prediction=model.predict([prepare("dog2.jpg")])
print(categories[int(prediction[0][0])])





