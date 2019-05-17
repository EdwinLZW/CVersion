#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"
import cv2
import numpy as np
from Model.BasicOperation import ImgBasicOperation


class ContourDetection(object):
    def __init__(self):
        self.basicop = ImgBasicOperation()

    def compare_difference_with_two_image(self, img1, img2):
        img1 = cv2.resize(self.basicop.acquire_image_info(img1, 0), (150, 200))
        img2 = cv2.resize(self.basicop.acquire_image_info(img2, 0), (150, 200))
        st2_1 = cv2.subtract(img1, img2)
        st2_2 = cv2.cvtColor(cv2.subtract(img1, img2), cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((11, 11), np.float32) / 25

        blur = cv2.GaussianBlur(st2_1, (11, 11), 0)  # 高斯模糊
        # dst = cv2.filter2D(st2_1, -1, kernel)

        # th2 = cv2.adaptiveThreshold(st2_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        ret, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        st2_22 = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
        ret2, threshold2 = cv2.threshold(st2_22, 75, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(threshold2, 300, 400)
        img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img2, contours, -1, (0, 0, 255), 1)
        cv2.imshow("st2_1", img2)
        cv2.waitKey(0)


if __name__ == '__main__':
    basicop = ContourDetection()
    basicop.compare_difference_with_two_image('img1.JPG', 'img2.JPG')