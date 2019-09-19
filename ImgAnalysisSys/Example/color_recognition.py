#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"

import cv2
import numpy as np


from BasicModel.BasicOperation import VideoBasicOperation


class ColorRecognition(VideoBasicOperation):
    def __init__(self):
        super(ColorRecognition, self).__init__()

    def identify_color(self, *args):
        self.cap = self.get_capture_video(0)
        if isinstance(self.cap, cv2.VideoCapture):
            while self.cap.isOpened():
                __ret,  __frame = self.cap.read()
                hsv = cv2.cvtColor(__frame, cv2.COLOR_BGR2HSV)
                lower_blue = np.array(args[0])
                upper_blue = np.array(args[1])
                cap_mask = cv2.inRange(hsv, lower_blue, upper_blue)
                res = cv2.bitwise_and(__frame, __frame, mask=cap_mask)
                cv2.imshow('res', res)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
            cv2.destroyAllWindows()


if __name__ == '__main__':
    core = ColorRecognition()
    core.identify_color([110, 50, 50], [130, 255, 255])


