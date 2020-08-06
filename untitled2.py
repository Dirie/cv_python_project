#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 12:43:00 2016

@author: dirie
"""
import sys
sys.path.append("/home/dirie/deep-learning/caffe/python")
import numpy as np

from scipy import signal
from scipy import linalg


import cv2
import pandas as pd
from matplotlib import pyplot as plt
import scipy.fftpack


import numpy as np
import cv2

def find_contours(img):
    #ret,thresh = cv2.threshold(img,127,255,0)
    im2, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    for cnt in contours:
#        x,y,w,h = cv2.boundingRect(cnt)
#        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    return img

cap = cv2.VideoCapture('jan28.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #f = find_contours(gray)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()












#a = np.array([[2,0,0,7,6,7,7],
#              [0,1,1,6,7,7,7],
#              [0,0,1,7,7,5,7],
#              [0,0,2,7,6,7,5],
#              [1,0,0,7,5,7,5],
#              [6,6,7,7,7,7,7],
#              [7,6,5,5,7,7,7]])
#f = np.array([[-0.25,-0.25,0],
#              [-0.25,0,0.25],
#              [0,0.25,0.25],])
#C = signal.convolve2d(a,f,boundary='fill')
#
#np.set_printoptions(threshold = np.nan)
#print(C)