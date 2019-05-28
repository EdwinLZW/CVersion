#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"

from collections import Counter
from Model.BasicOperation import ImgBasicOperation


class OSTUAlgorithm(object):
    def __init__(self, filename):
        self._img = ImgBasicOperation().acquire_image_info(filename, 3)
        print self._img.shape
        self._gray_level = {}

    def ostu_one_algorithm(self):
        U, U0, W0, U1, W1 ,G= 0, 0, 0, 0, 0, 0
        __pixel_Y, __pixel_X = self._img.shape
        __total_pixel = __pixel_Y*__pixel_X
        __gray = Counter(self._img.flatten()).keys()
        __propo = [float(i) / __total_pixel for i in Counter(self._img.flatten()).values()]
        for i in range(len(__gray)):
            U += __gray[i]*__propo[i]
        for t in __gray:
            for j in range(__gray.index(t)):
                U0 += __gray[j]*__propo[j]
                W0 += __propo[j]
            U1 = U - U0
            W1 = 1 - W0
            G = W0*pow(U-U0, 2)+W1*pow(U-U1, 2)
            self._gray_level.setdefault(t, round(G, 2))
        return max(self._gray_level, key=self._gray_level.get)


if __name__ == '__main__':
    sss = OSTUAlgorithm("Lena.jpg")
    print sss.ostu_one_algorithm()

