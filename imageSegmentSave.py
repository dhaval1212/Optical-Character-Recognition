import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img = cv2.imread('myTestImages/h.jpg', cv2.IMREAD_GRAYSCALE)

img = cv2.GaussianBlur(img,(3,3), 0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 13, 10)
plt.imshow(img)
image, contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
d=0
string = ""
filename = "myTestImages/h"
no = 100
for ctr in contours:
    # Get bounding box
    
    x, y, w, h = cv2.boundingRect(ctr)
        #print(x, y, w, h)
    if h > 4 and w > 2:
        # Getting ROI
        roi = image[y-1:y+h+1, x-3:x+w+3]
        roi = cv2.resize(roi, (28,28))
        
        for i in range(len(roi)):
            for j in range(len(roi[0])):
                roi[i][j] = 255 - roi[i][j]
                
        no += 1
        cv2.imwrite(filename+str(no)+".jpg",roi)