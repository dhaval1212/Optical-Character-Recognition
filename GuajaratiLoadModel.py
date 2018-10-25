# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 13:12:37 2018

@author: Dhaval
"""

import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt

CATEGORIES = ["A","BA","BHA","CA","CHA","DA","DDA","DDHA_in","DHA","faa","GA","GHA_in",
              "GNA_in","HA","I","II","JA","JHA","KA","KHA","KSHA_in","LA","LA_M","LLA",
              "MA","NA","NNA","PA","PHA_in","RA","S_SHRA","SA","SHA","SSA","TA","THA",
              "TTA","TTHA_in","U","UU_in","VA","YA"]

test_image = cv2.imread('GujaratiTestImages/dha_in.png',cv2.IMREAD_GRAYSCALE)

test_image = cv2.GaussianBlur(test_image, (3,3), 0)
#test_image = cv2.adaptiveThreshold(test_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,
 #           cv2.THRESH_BINARY_INV, 7,10)
plt.imshow(test_image)

test_image = cv2.resize(test_image, (28,28))
for i in range(len(test_image)):
    for j in range(len(test_image[0])):
        test_image[i][j] = 255 - test_image[i][j]
plt.imshow(test_image)

test_image = np.expand_dims(test_image, axis = 0)
test_image = np.expand_dims(test_image, axis = 3)

test_image = test_image.reshape(test_image.shape[0], 28, 28, 1).astype('float32')

test_image = test_image / 255

#print(test_image.shape)

#loading saved model
new_model = tf.keras.models.load_model('GujaratiOCRModel')

predictions = new_model.predict(np.array(test_image))
print("Predicted character is :",end = " ")
print(CATEGORIES[np.argmax(predictions[0])])

img = cv2.resize(test_image[0], (28,28))
plt.imshow(img)
