import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

CATEGORIES = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
              "Q","R","S","T","U","V","W","X","Y","Z"]
new_model = tf.keras.models.load_model('Thresholded_Blurred_Trained')


img = cv2.imread('myTestImages/string/shridhar.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img, (3, 3), 0)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 13, 10)
plt.imshow(img)
image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
d=0
string = ""
for ctr in contours:
    # Get bounding box
    
    x, y, w, h = cv2.boundingRect(ctr)
        #print(x, y, w, h)
    if h > 4 and w > 2:
        # Getting ROI
        roi = image[y-1:y+h+1, x-1:x+w+2]
        roi = cv2.resize(roi, (28,28))
        
        #pass the data into model
        roi = np.expand_dims(roi, axis = 0)
        roi = np.expand_dims(roi, axis = 3)
        
        roi = roi.reshape(roi.shape[0],28,28,1).astype('float32')
        
        roi = roi / 255
        
        predictions = new_model.predict(np.array(roi))
        string += CATEGORIES[np.argmax(predictions[0])]

#roi = image[25:47, 199:216]
#cv2.imwrite("s65.jpg",roi)
#plt.imshow(roi)
    
print("Predicted string is : " ,string)
