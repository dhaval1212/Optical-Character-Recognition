import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
CATEGORIES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

test_image = cv2.imread('myTestImages/s65.jpg',cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, (28,28))
"""for i in range(len(test_image)):
    for j in range(len(test_image[0])):
        test_image[i][j] = 255 - test_image[i][j] """
#plt.imshow(test_image)

test_image = np.expand_dims(test_image, axis = 0)
test_image = np.expand_dims(test_image, axis = 3)

#print(test_image.shape) 

#loading saved model
new_model = tf.keras.models.load_model('Thresholded_Blurred_Trained')

predictions = new_model.predict(np.array(test_image))
print("Predicted character is :",end = " ")
print(CATEGORIES[np.argmax(predictions[0])])

img = cv2.resize(test_image[0], (28,28))
plt.imshow(img)
