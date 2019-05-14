#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Simon Liu"

import os
import cv2
import matplotlib.pyplot as plt

IMG_FILE_PATH = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'Templete')


class ImgBasicOperation(object):
    def __init__(self):
        self.img_list = []
        self.point_bgr = None

    def acquire_image_info(self, filename, flag=False):
        """
        -- cv2.imread('../Peppa.jpeg', cv2.IMREAD_UNCHANGED) --read image: there are two para, first is image`s path,
                                            second is to show how to read image the second always include :
                                            cv2.IMREAD_COLOR  read color images，
                                            cv2.IMREAD_GRAYSCALE  read images in grayscale mode，
                                            cv2.IMREAD_UNCHANGED  read an image and include the alpha channel of the image

        :param filename: image`s path
        :param flag:
        :return:
        """
        img = cv2.imread(self.getfilepath(filename), cv2.IMREAD_UNCHANGED)
        if flag:
            return img.shape
        else:
            # self.put_text_to_image(img)
            return img

    def read_point_color_value(self, filename, x, y):
        img_info = self.acquire_image_info(filename)
        self.point_bgr = img_info[y, x]
        return self.point_bgr

    def add_sub_image(self, bg, *args):
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
                i +=100
            return bg_obj
        else:
            return 'ERROR -- no bg image!'

    def put_text_to_image(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text='col=width=X0,row=height-Y0', org=(0, 0), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2, bottomLeftOrigin=True)  # text,
        cv2.putText(img, text='col=width=X10,row=height-Y30', org=(10, 30), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2)  # text,
        cv2.putText(img, text='col=width=X100,row=height-Y300', org=(100, 300), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2)  # text,
        cv2.putText(img, text='col=width-X300,row=height-Y100', org=(300, 100), fontFace=font, fontScale=0.5, color=(0, 255, 0), thickness=2)  # text,

    def save_image(self, savefile, img):
        """
        -- cv2.imwrite(savefile, img) save image
        :param savefile:
        :param img:
        :return:
        """
        cv2.imwrite(self.getfilepath(savefile), img)

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

    def matplotlib_show_image(self, filename):
        "There is a problem, when color image load with opencv ,the model is BGR, but the model is RGB with matplotlib."
        img = self.acquire_image_info(filename)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img2)
        plt.show()

    def getfilepath(self, filename):
        return '{}/{}'.format(IMG_FILE_PATH, filename)


class VideoBasicOperation(object):
    def __init__(self):
        self.cap = None
        self.fourcc = cv2.cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.out_video_path = cv2.VideoWriter(IMG_FILE_PATH + '/output.avi', fourcc, 20.0, (640, 480))
        self.init_capture()

    def init_capture(self, flag=0):
        try:
            cap = cv2.VideoCapture(flag)
            if cap.isOpened():
                self.cap = cap
            else:
                cap.open('')
        except Exception as e:
            raise e

    def acquire_capture(self):
        if isinstance(self.cap, cv2.VideoCapture):
            while cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 0)
                    out.write(frame)
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # imgbasicop = ImgBasicOperation()
    # print imgbasicop.acquire_image_info('Peppa.jpeg', flag=False)
    # print imgbasicop.read_point_color_value('Peppa.jpeg', 300, 500)
    # img = imgbasicop.add_sub_image('Peppa.jpeg', 'Lena.jpg', 'opencv_logo.png')
    # imgbasicop.save_image('test.png', img)
    # imgbasicop.show_image('test', img)
    # imgbasicop.matplotlib_show_image('box.png')

    # sss = VideoBasicOperation()

    cap = cv2.VideoCapture(0)
    print cap
    fourcc = cv2.cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(IMG_FILE_PATH +'/output.avi', fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', gray)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

