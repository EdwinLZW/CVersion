#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

IMG_FILE_PATH = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'Templete')

'''
img = cv2.imread('cute.jpg',0)
plt.imshow(img,cmap='gray',interpolation='bicubic')
plt.xticks([],plt.yticks([]))  # to hide tick values on X and Y axis
plt.show()
'''

'''
Color image loaded by OpenCV is in BGR mode.
But Matplotlib displays in RGB mode.
So color images will not be displayed correctly in Matplotlib if image is read with OpenCV.
Please see the exercises for more details.
'''
img = cv2.imread('{}/{}'.format(IMG_FILE_PATH, 'Lena.jpg'))
# b, g, r = cv2.split(img)
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img2 = img[:,:,::-1]    this can be faster
plt.subplot(121)
plt.imshow(img)  # expects distorted color
plt.subplot(122)
plt.imshow(img2)  # expects true color
plt.show()

cv2.imshow('bgr image', img)  # expects true color
cv2.imshow('rgb image', img2)  # expects distrorted color
cv2.waitKey(0)
cv2.destroyAllWindows()
