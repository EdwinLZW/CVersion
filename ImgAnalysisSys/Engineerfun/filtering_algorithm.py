#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"

import cv2
import numpy as np
from Model.BasicOperation import ImgBasicOperation
from pandas import DataFrame
import matplotlib.pyplot as plt
import math

kernel_A = np.array(
    [[0.1, 0.1, 0.1],
     [0.1, 0.2, 0.1],
     [0.1, 0.1, 0.1]]
)

kernel_B = np.ones((5, 5), dtype=np.float32) / 25.0

kernel_C = np.array(
    [[(-1-1), (-10), (-11)],
     [(0-1), (00), (01)],
     [(1-1), (10), (11)]]
)
DataFrame(kernel_C).to_csv('t1.csv')


class FilteringAlgorithm(object):
    def __init__(self):
        self._kernel_A = kernel_A
        self._kernel_B = kernel_B
        self._getpic = ImgBasicOperation()

    def boxbluralg(self, filename, normalize=True, edge=False):
        """
        方框滤波 - 自己设置卷积核，边缘不处理
        :param filename:
        :param normalize:
        :param edge:
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __ls = []
        __row_arry = []
        for i in range(self._kernel_A.shape[1]/2, __img.shape[1] - self._kernel_A.shape[1]/2):
            for j in range(self._kernel_A.shape[0]/2, __img.shape[0] - self._kernel_A.shape[0]/2):
                __value = 0
                for k in range(self._kernel_A.shape[0]):
                    for m in range(self._kernel_A.shape[1]):
                        if normalize:
                            __value += __img[i - 1 + k][j - 1 + m] * self._kernel_A[k][m]
                        else:
                            __value += __img[i - 1 + k][j - 1 + m]
                            if __value > 255:
                                __value = 255
                __ls.append(round(__value, 0))
            __row_arry.append(__ls)
            __ls = []
        return DataFrame(np.array(__row_arry))

    def medianbluralg(self, filename, kernel_sizer_adius=1, edge=False):
        """
        中值滤波 3*3
        :param filename:
        :param kernel_sizer_adius:
        :param edge: 是否带边缘处理
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __ls = []
        __row_arry = []
        for i in range(kernel_sizer_adius, __img.shape[1] - kernel_sizer_adius):
            for j in range(kernel_sizer_adius, __img.shape[0] - kernel_sizer_adius):
                __value = []
                for k in range(kernel_sizer_adius*2+1):
                    for m in range(kernel_sizer_adius*2+1):
                        __value.append(__img[i - kernel_sizer_adius + k][j - kernel_sizer_adius + m])
                __value.sort()
                __ls.append(__value[len(__value)/2])
            __row_arry.append(__ls)
            __ls = []
        return DataFrame(np.array(__row_arry))

    def guassianmodel(self, kernel_sizer_adius, sigma):
        __row = []
        __column = []
        proportion_sum = 0
        for i in range(-kernel_sizer_adius, kernel_sizer_adius+1):
            for j in range(-kernel_sizer_adius, kernel_sizer_adius + 1):
                __guass_model = math.exp(-(pow(i, 2) + pow(j, 2))/(2.0 * pow(sigma, 2)))/(2.0 * math.pi * pow(sigma, 2))
                proportion_sum += __guass_model
                __column.append(__guass_model)
            __row.append(__column)
            __column = []
        return np.array(__row).transpose()/proportion_sum

    def guassianbluralg(self, filename, kernel_sizer_adius=1, sigma=0.849):
        """
        中值滤波 3*3
        :param filename:
        :param args:
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __guassian_arry = self.guassianmodel(kernel_sizer_adius, sigma)
        __ls = []
        __row_arry = []

        for i in range(__guassian_arry.shape[1] / 2, __img.shape[1] - __guassian_arry.shape[1] / 2):
            for j in range(__guassian_arry.shape[0] / 2, __img.shape[0] - __guassian_arry.shape[0] / 2):
                __value = 0
                for k in range(__guassian_arry.shape[0]):
                    for m in range(__guassian_arry.shape[1]):
                        __value += __img[i - 1 + k][j - 1 + m] * __guassian_arry[k][m]
                __ls.append(round(__value, 0))
            __row_arry.append(__ls)
            __ls = []
        return DataFrame(np.array(__row_arry))

    def opencv_blur(self, filename, kernel_sizer_adius):
        """
        cv2.blur(img, (3, 3))  均值滤波
        :param filename:
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __blur = cv2.blur(__img, (kernel_sizer_adius, kernel_sizer_adius))
        return __blur

    def opencv_boxfilter(self, filename, kernel_sizer_adius):
        """
        cv2.boxfilter(img, -1, (3, 3), normalize=True) 方框滤波，
        参数说明当normalize=True时，与均值滤波结果相同， normalize=False，表示对加和后的结果不进行平均操作，大于255的使用255表示
        :param filename:
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __boxblur = cv2.boxFilter(__img, -1, (kernel_sizer_adius, kernel_sizer_adius), normalize=True)
        return __boxblur

    def opencv_medianBlur(self, filename, kernel_sizer_adius):
        """
        cv2.medianBlur(img, 3)    中值滤波，相当于将9个值进行排序，取中值作为当前值
        参数说明：img表示当前的图片，3表示当前的方框尺寸在图像的读取中，会存在一些躁声点，如一些白噪声，因此我们需要进行去躁操作
        :param filename:
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __blur = cv2.medianBlur(__img, kernel_sizer_adius)
        return __blur

    def opencv_Guassianblur(self, filename, kernel_sizer_adius):
        """
        cv2.Guassianblur(img, (3, 3), 1) 高斯滤波，
        参数说明: 1表示σ， x表示与当前值得距离，计算出的G(x)表示权重值
        :param filename:
        :return:
        """
        __img = self._getpic.acquire_image_info(filename, 3)
        __blur = cv2.GaussianBlur(__img, (kernel_sizer_adius, kernel_sizer_adius), 1)
        return __blur


if __name__ == '__main__':
    boxblur = FilteringAlgorithm()
    plt.subplot(221), plt.title("blur"), plt.imshow(boxblur.opencv_blur('Peppa.jpeg', 9))
    plt.subplot(222), plt.title("boxfilter"), plt.imshow(boxblur.opencv_boxfilter('Peppa.jpeg', 9))
    plt.subplot(223), plt.title("medianBlur"), plt.imshow(boxblur.opencv_medianBlur('Peppa.jpeg', 9))
    plt.subplot(224), plt.title("Guassianblur"), plt.imshow(boxblur.opencv_Guassianblur('Peppa.jpeg', 9))
    plt.show()











