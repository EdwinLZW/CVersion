#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Zhouwu Liu"
__copyright__ = "Copyright 2019, PRMeasure Inc."
__email__ = "zhouwu.liu@prmeasure.com"

import cv2
import numpy as np
import matplotlib.pyplot as plt


class ContrastEnhance(object):
    def __init__(self, filename):
        self.img = cv2.imread(filename, 2)

    def calcgrayhist(self):
        """
        灰度直方图
        :return:
        """
        height, width = self.img.shape[:2]
        grayHist = np.zeros([256], np.uint32)
        for i in range(height):
            for j in range(width):
                grayHist[self.img[i, j]] += 1
        return grayHist

    def calcnormalize(self):
        """
        直方图正规化原理： （O(r,c)-Omin）/（I(r,c)-Imin）= （Omax-Omin）/（Imax-Imin）
        :return:
        """
        Imin, Imax = cv2.minMaxLoc(self.img)[:2]
        print cv2.minMaxLoc(self.img)[:2]
        Omax, Omin = 255, 0
        a = (Omax-Omin)*1.0/(Imax-Imin)
        b = Omin-a*Imin
        dst = a * self.img + b   # float的值
        dst = dst.astype(np.uint8)    # int的值
        return dst

    def gammatransform(self, gamma=0.5):
        """
        伽马变换
        :param gamma:
        :return:
        """
        img = self.img/255.0
        dst = np.power(img, gamma)*255
        dst = dst.astype(np.uint8)
        return dst

    def calcequalhist(self):
        """
        直方图均衡化：
            1、计算图像的灰度直方图
            2、计算灰度直方图的累加直方图
            3、根据1，2步找到输入与输出的映射关系
            4、得到均衡化以后的灰度值
        :return:
        """
        grayhist = self.calcgrayhist()
        cumuhist = np.zeros(256, np.float32)
        total_piexl = float(np.sum(grayhist))
        normal_grayhist = grayhist / total_piexl
        equalhistimg = np.zeros((self.img.shape), np.uint8)
        for i in range(256):
            if i == 0:
                cumuhist[i] = normal_grayhist[0]
            else:
                cumuhist[i] = cumuhist[i-1] + normal_grayhist[i]
        cumuhist = (cumuhist*255).astype(np.uint8)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                equalhistimg[i, j] = cumuhist[self.img[i, j]]
        return equalhistimg


if __name__ == '__main__':
    CE = ContrastEnhance('test.jpg')
    grayhist= CE.calcequalhist()
    cv2.imshow("dst", grayhist)
    cv2.waitKey(0)
