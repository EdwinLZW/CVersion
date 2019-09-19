#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"


import os
import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

TEMPLETE_FILE_PATH = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'Templete')

IMAGE_FORMAT = ('BMP', 'DIB', 'JPEG', 'JPG', 'JPE', 'PNG', 'PBM', 'PGM', 'PPM', 'SR', 'RAS', 'TIFF', 'TIF', 'EXR', 'JP2')
VIDEO_FORMAT = ('AVI', 'MPG', 'MPE', 'MPEG', 'DAT', 'VOB', 'ASF', '3PG', 'MP4', 'RMVB', 'FLV', 'MOV')


class ImgBasicOperation(object):

    def __init__(self):
        self.img_list = []
        self.point_bgr = None

    @classmethod
    def getfilepath(cls, imagefile):
        if isinstance(imagefile, str) and '.' in imagefile:
            img_format = imagefile.split('.')[1].upper()
            if img_format.upper() in IMAGE_FORMAT:
                return '{}/{}'.format(TEMPLETE_FILE_PATH, imagefile)
            else:
                raise ImportError, 'openCV can`t deal with this imagefile format!'
        else:
            raise ImportError, 'Image file input ERROR!'

    @classmethod
    def parse_imageread_model(cls, modelflag):
        """
        - cv2.IMREAD_COLOR:      read color images，
        - cv2.IMREAD_GRAYSCALE:  read images in grayscale mode，
        - cv2.IMREAD_UNCHANGED:  read an image and include the alpha channel of the image
        :param modelflag: 0,1,2
        :return: openCV image read_model
        """
        if modelflag == 0:
            return cv2.IMREAD_UNCHANGED
        elif modelflag == 1:
            return cv2.IMREAD_COLOR
        elif modelflag == 2:
            return cv2.IMREAD_GRAYSCALE
        else:
            return 0

    def get_image_size(self, filename):
        """
        :param filename:
        :return: (row/height/Y, column/width/X, CH)
        """
        return self.acquire_image_info(filename).shape

    def read_point_color_value(self, filename, *args):
        img_info = self.acquire_image_info(filename)
        self.point_bgr = img_info[args]
        return self.point_bgr

    def acquire_image_info(self, filename, flag=0):
        """
        :param filename: image`s path
        :param flag: read_Model 0-cv2.IMREAD_UNCHANGED,1-IMREAD_COLOR,3-IMREAD_GRAYSCALE
        :return: image obj(numpy.ndarray)
        """
        img = cv2.resize(cv2.imread(self.getfilepath(filename), self.parse_imageread_model(flag)), (300, 300))
        if img is not None:
            # print cv2.split(img)[0]
            return img
        else:
            raise ImportError, 'Image File read ERROR!'

    def alter_image_size(self, filename, *args):
        """

        :param filename: image file
        :param args: height/Y = args[0], width/X = args[1]
        :return:
        """
        if len(args) != 2:
            raise TypeError,  "function takes exactly 2 arguments ({} given)".format(len(args))
        return cv2.resize(self.acquire_image_info(filename), args)

    def save_image(self, savefilepath, img):
        """
        -- cv2.imwrite(savefile, img) save image
        :param savefile:
        :param img:
        :return:
        """
        cv2.imwrite(self.getfilepath(savefilepath), img)

    def show_image(self, titlename, imgname):
        """
        -- cv2.imshow('images', img) show picture ,named 'images'
        -- cv2.waitKey() Keyboard binding function
        :param titlename:
        :param imgname:
        :return:
        """
        cv2.imshow(titlename, imgname)
        cv2.moveWindow(titlename, x=imgname.shape[0], y=0)
        cv2.waitKey(0)
        if cv2.waitKey(0) == 27:
            cv2.destroyWindow(titlename)

    def add_sub_image(self, bg, *args):
        """
        The Function main use to paste sub_image to background, just replace the narray
        :param bg:       background image
        :param args:
        :return:
        """
        i = 50
        sub_img = []
        if len(args) > 0:
            for filename in args:
                adjust_img = cv2.resize(self.acquire_image_info(filename), (40, 40))
                sub_img.append(adjust_img)
        if bg:
            bg_obj = self.acquire_image_info(bg)
            for img_file in sub_img:
                bg_obj[200+i:200+i + img_file.shape[0], 300+i:300+i + img_file.shape[1]] = img_file[:, :, 0:3]
                i += 100
            return bg_obj
        else:
            return 'ERROR -- no bg image!'

    def put_text_to_image(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text='col=width=X0,row=height-Y0', org=(0, 0), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2, bottomLeftOrigin=True)  # text,
        cv2.putText(img, text='col=width=X10,row=height-Y30', org=(10, 30), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2)  # text,
        cv2.putText(img, text='col=width=X100,row=height-Y300', org=(100, 300), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2)  # text,
        cv2.putText(img, text='col=width-X300,row=height-Y100', org=(300, 100), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2)  # text,

    def matplotlib_show_image(self, filename):
        "There is a problem, when color image load with opencv ,the model is BGR, but the model is RGB with matplotlib."
        img = self.acquire_image_info(filename)
        # img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    def singel_pixel_modify(self, filename, y, x, ch, pixel):
        img = self.acquire_image_info(filename)
        # img.itemset((y, x, ch), pixel)
        print cv2.split(img)


class VideoBasicOperation(object):
    def __init__(self):
        self.cap = None
        self.fourcc = cv2.cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # define video fourcc(Four-Character Codes), 32bit

    @classmethod
    def get_capture_video(cls, flag=None):
        """
        :param flag:  flag indicate device, 0 - local camera, other - device or video file
        :return: VideoCapture-obj
        """
        try:
            cap = cv2.VideoCapture(flag)
            time.sleep(1)
            if cap.isOpened():
                return cap
            else:
                cap.open('')
        except Exception as e:
            raise e

    @classmethod
    def getfilepath(cls, videofile):
        if isinstance(videofile, str) and '.' in videofile:
            img_format = videofile.split('.')[1].upper()
            if img_format.upper() in VIDEO_FORMAT:
                return '{}/{}'.format(TEMPLETE_FILE_PATH, videofile)
            else:
                raise ImportError, 'openCV can`t deal with this videofile format!'
        else:
            raise ImportError, 'Video file input ERROR!'

    def capture_and_save_video(self, outvideo):
        """
        use camare to  capture vedio and save video.
        :param outvideo:   the saving video file name.
        :return:
        """
        self.cap = self.get_capture_video(flag=0)
        if isinstance(self.cap, cv2.VideoCapture):
            out_video = cv2.VideoWriter(self.getfilepath(outvideo), self.fourcc, 20.0, (120, 150))
            # width_ret = self.cap.set(3, 320)
            # height_ret = self.cap.set(4, 480)
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    video_obj = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out_video.write(video_obj)
                    cv2.imshow('{}'.format(outvideo), video_obj)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            self.cap.release()
            out_video.release()
            cv2.destroyAllWindows()

    def display_video(self, videofile):
        """
        display local vedio.
        :param videofile: video name to be displayed.
        :return:
        """
        self.cap = self.get_capture_video(self.getfilepath(videofile))
        # print self.cap.read()[1]
        # print self.cap.get(cv2.CAP_PROP_FPS)                # get Frame_rate
        # print self.cap.get(cv2.CAP_PROP_FRAME_COUNT)        # get total Frame_rate
        # print self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)       # get video height
        # print self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)        # get video width
        # print self.cap.get(cv2.CAP_PROP_POS_FRAMES)         # get now Frame_rate
        # print self.cap.get(cv2.CAP_PROP_MODE)               # get the current capture mode
        if isinstance(self.cap, cv2.VideoCapture):
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    video_obj = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('{}'.format(videofile), video_obj)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            self.cap.release()
            cv2.destroyAllWindows()


class DrawingBasicOperation(object):
    def __init__(self):
        self.bg_img = np.zeros((512, 512, 3), np.uint8)

    @classmethod
    def getfilepath(cls, videofile):
        if isinstance(videofile, str) and '.' in videofile:
            img_format = videofile.split('.')[1].upper()
            if img_format.upper() in IMAGE_FORMAT:
                return '{}/{}'.format(TEMPLETE_FILE_PATH, videofile)
            else:
                raise ImportError, 'openCV can`t deal with this videofile format!'
        else:
            raise ImportError, 'Video file input ERROR!'

    @classmethod
    def show_window(cls, bgimg):
        winname = 'example'
        cv2.namedWindow(winname, 0)
        cv2.imshow(winname, bgimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawline(self):
        # cv2.line(self.bg_img, (0, 0), (511, 511), (255, 0, 0), 5)       # draw line
        pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.bg_img, [pts], True, (255, 255, 0), 8)
        self.show_window(self.bg_img)

    def drawrectangle(self):
        cv2.rectangle(self.bg_img, (384, 0), (510, 128), (0, 255, 0), 3)
        self.show_window(self.bg_img)

    def drawcircle(self):
        cv2.circle(self.bg_img, center=(447, 63), radius=63, color=(0, 0, 255), thickness=-1)
        self.show_window(self.bg_img)

    def drawellipse(self):
        cv2.ellipse(self.bg_img, center=(256, 256), axes=(100, 50), angle=0, startAngle=0, endAngle=180, color=255,
                    thickness=-1)
        self.show_window(self.bg_img)

    def puttext_to_image(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.bg_img, text='Simon', org=(10, 400), fontFace=font, fontScale=1, color=(255, 255, 255), thickness=1,bottomLeftOrigin=True)#text, org, fontFace, fontScale, color, thickness=
        cv2.putText(self.bg_img, text='OpenCV', org=(10, 500), fontFace=font, fontScale=4, color=(255, 255, 255), thickness=2) #text, org, fontFace, fontScale, color, thickness=
        self.show_window(self.bg_img)


class DataOperation(object):
    def __init__(self):
        pass

    def write_data_to_csv(self, data, filename):
        if isinstance(data, np.ndarray):
           DataFrame(data).to_csv(filename)

