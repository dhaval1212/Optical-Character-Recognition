# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 16:18:23 2018

@author: Dhaval
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.model_selection import train_test_split
from keras.utils import np_utils


DATADIR = "E:\\Datasets\\Alphabets"

CATEGORIES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

IMG_SIZE = 28

#create training data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)  # create path to dataset
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)):  # iterate over each image per alphbets
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  #size to normalize data size
                for i in range(len(img)):
                    for j in range(len(img[0])):
                        img[i][j] = 255 - img[i][j]
                training_data.append([img, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
            
#loading data from dataset  folders
"""
create_training_data()


print(len(training_data))

pickle_out = open("training_data.pickle","wb")
pickle.dump(training_data, pickle_out)
pickle_out.close()
"""

#loading data from pickle file
pickle_in = open("training_data.pickle","rb")
training_data = pickle.load(pickle_in)


#randomize data
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

seed = 785
# split the data into training (50%) and testing (50%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.30, random_state=seed)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

num_classes = Y_test.shape[1]


#model training
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=200, verbose=2)


#model evaluation
val_loss, val_acc = model.evaluate(X_test, Y_test)
print(val_loss)
print(val_acc)

#save the model
model.save('OCRModel')
#load saved model
new_model = tf.keras.models.load_model('OCRModel')

#predict image on test data
predictions = new_model.predict(X_test)

print(np.argmax(predictions[1000]))

img = cv2.resize(X_test[1000], (28,28))

plt.imshow(img)

