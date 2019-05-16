#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"

from Model.BasicOperation import *


class AddLogoToImage(object):
    def __init__(self):
        self._img_obj = ImgBasicOperation()

    def addlogo(self, bgimg, logoimg, *args):
        if len(args) == 4:
            __bg_img = self._img_obj.acquire_image_info(bgimg)
            __logo_img = cv2.resize(self._img_obj.acquire_image_info(logoimg, 1), (args[0], args[1]))
            rows, clos, channel = __logo_img.shape
            roi = __bg_img[__bg_img.shape[0] - rows:__bg_img.shape[0], __bg_img.shape[0] - clos:__bg_img.shape[0]]

            logo_img_gray = cv2.cvtColor(__logo_img, cv2.COLOR_BGR2GRAY)
            ret, logo_img_mask = cv2.threshold(logo_img_gray, args[2], args[3], cv2.THRESH_BINARY)
            banarynot_logo_img = cv2.bitwise_not(logo_img_mask)

            banary_roi = cv2.bitwise_and(roi, roi, mask=logo_img_mask)
            banaryand_logo_img = cv2.bitwise_and(__logo_img, __logo_img, mask=banarynot_logo_img)

            dst = cv2.add(banary_roi, banaryand_logo_img)
            __bg_img[__bg_img.shape[0] - rows:__bg_img.shape[0], __bg_img.shape[0] - clos:__bg_img.shape[0]] = dst

            cv2.imshow('addlogo', __bg_img)
            cv2.waitKey(0)
        else:
            raise ImportError, 'Logo size imput ERROR!'


if __name__ == '__main__':
    addlog = AddLogoToImage()
    addlog.addlogo('Lena.jpg', 'opencv_logo.png', 120, 120, 150, 255)
