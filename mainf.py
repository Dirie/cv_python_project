# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 19:10:04 2016

@author: dirie
"""

import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib as plt
#import cv2

import numpy as np
import cv2
#/home/dirie/anaconda3/share/OpenCV/haarcascades
face_cascade = cv2.CascadeClassifier('/home/dirie/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture('jan28.avi')
#cap = cv2.VideoCapture(0)
while(cap.isOpened()):
#while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



