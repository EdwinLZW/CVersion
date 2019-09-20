#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"
import cv2
import numpy as np


class AffineTransformation(object):
    """
    仿射变换
    """
    def __init__(self, filename):
        self.__filename = filename

    def linear_move(self):
        """
        平移
        :return:
        """
        img = cv2.imread(self.__filename)
        H = np.float32([[1, 0, 100], [0, 1, 50]])
        rows, cols = img.shape[:2]
        res = cv2.warpAffine(img, H, (rows, cols))
        return res

    def shrink(self):
        """
        缩放
        :return:
        """
        img = cv2.imread(self.__filename)
        height, width = img.shape[:2]
        res2 = cv2.resize(img, (0.5 * width, 0.5 * height), interpolation=cv2.INTER_CUBIC)
        return res2

    def rotato(self):
        """
        旋转
        :return:
        """
        img = cv2.imread(self.__filename)
        rows, cols = img.shape[:2]
        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
        # 第三个参数：变换后的图像大小
        res = cv2.warpAffine(img, M, (rows, cols))
        return res

    def affinetransform(self):
        """
        仿射变换
        :return:
        """
        img = cv2.imread(self.__filename)
        rows, cols = img.shape[:2]
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)
        res = cv2.warpAffine(img, M, (rows, cols))
        return res

    def nearest_interploat(self, dsize):
        """
        最近邻插值(适应灰度图)
        :param self
        :param dsize: 目标尺寸
        :return: 目标图片
        """
        src_img = cv2.imread(self.__filename)
        height, width, channels = src_img.shape
        emptyImage = np.zeros((dsize), np.uint8)
        sh = height*1.0 / dsize[0]
        sw = width*1.0 / dsize[1]
        for i in range(dsize[0]):
            for j in range(dsize[1]):
                x = int(round(i * sh))
                y = int(round(j * sw))
                emptyImage[i, j] = src_img[x, y]
        return emptyImage

    def bilinear_interpolation(self, dsize):
        """
        双线性插值
        :param out_dim:
        :return:
        """
        src_img = cv2.imread(self.__filename)
        height, width, channels = src_img.shape
        emptyImage = np.zeros((dsize), np.uint8)
        sh = height * 1.0 / dsize[0]
        sw = width * 1.0 / dsize[1]
        for i in range(dsize[0]):
            for j in range(dsize[1]):
                x = int(i * sh)
                y = int(j * sw)
                p = round(i * sh, 3) - x
                q = round(j * sw, 3) - y
                if x < height-1 and y < width-1:
                    emptyImage[i, j] = (1-p)*((1-q)*src_img[x, y] + q*src_img[x, y+1]) + p*((1-q)*src_img[x+1, y] + q*src_img[x+1, y+1])
                elif x == height-1 and y != width-1:
                    emptyImage[i, j] = src_img[x, y]
                elif x != height-1 and y == width-1:
                    emptyImage[i, j] = src_img[x, y]
                else:
                    emptyImage[i, j] = src_img[height-1, width-1]
        return emptyImage


class ProjectionTransformation(object):
    """
    投影变换
    """
    def __init__(self, filename):
        self.__filename = filename

    def p_transform(self):
        img = cv2.imread(self.__filename)
        h, w = img.shape[:2]
        src = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], np.float32)
        dst = np.array([[50, 50], [w/3, 50], [50, h-1], [w-1, h-1]], np.float32)
        P = cv2.getPerspectiveTransform(src, dst)  # 计算投影矩阵
        r = cv2.warpPerspective(img, P, (w, h), borderValue=0)
        return r


class PolarCoordinateTransformation(object):
    """
    极坐标变换
    """
    def __init__(self, filename):
        self.__filename = filename

    def polar_transform(self):
        pass


if __name__ == '__main__':
    aff = ProjectionTransformation("circle1.jpg")
    ni = aff.p_transform()
    cv2.imshow('dst', ni)
    cv2.waitKey(0)


