import numpy as np
import cv2
from matplotlib import pyplot as plt

image = cv2.imread('852x622_.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(grayImage, 1.03, 5)

body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
body = body_cascade.detectMultiScale(grayImage, 1.01, 10)

for (x,y,w,h) in body:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
    
    body_image_gray = grayImage[y:y+h, x:x+w]
    body_image_color = image[y:y+h, x:x+w]
    
    faces_in_body = face_cascade.detectMultiScale(body_image_gray)

    for (xf,yf,wf,hf) in faces_in_body:
        cv2.rectangle(body_image_color,(xf,yf),(xf+wf,yf+hf),(0,255,0),2)
        
        
plt.figure(figsize=(12,12))
plt.imshow(image)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()