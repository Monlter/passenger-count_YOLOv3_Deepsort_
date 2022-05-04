#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default="./1.mp4")
ap.add_argument("-c", "--class",help="name of class", default="person")
args = vars(ap.parse_args())


class YOLO(object):
    def __init__(self):
        #yolo_tiny
        # self.model_path = 'model_data/yolo-tiny.h5'  # yolo_tiny权重文件
        # self.anchors_path = 'model_data/tiny_yolo_anchors.txt' #yolo_tiny  6个anchor box
        #yolo
        self.model_path = 'model_data/yolo.h5'   #权重文件(原权重文件（种类为80）model_data/yolo.h5)
        self.anchors_path = 'model_data/yolo_anchors.txt'    #anchor box 9个，从小到大排列  13*13、26*26、52*52feature map  特征图越小，感受域越大，对大目标越敏感
        self.classes_path = 'model_data/coco_classes.txt'    #原权重文件（种类为80）model_data/coco_classes.txt)类别数
        self.score = 0.5        #score置信度阈值，小于阈值被删除
        self.iou = 0.5          #iou阈值，大于阈值的重叠框被删除
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)     不同于这个尺寸的输入会先调整到标准大小
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()        #由generate（）函数完成目标检测

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)   #转换成用户目录
        with open(classes_path) as f:
            class_names = f.readlines()         #读取所有行并返回列表
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        """

        :return: boxes,scores,classes
        """
        model_path = os.path.expanduser(self.model_path)         #获取model路径
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'   #判断model是否以h5结尾

        self.yolo_model = load_model(model_path, compile=False)       #下载model ###################################################################question-1:load_model()
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.生成绘制边框的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)        #h(色调）：x/len(self.class_names)  s(饱和度）：1.0  v(明亮）：1.0
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))       #hsv转换为rgb
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))            #通过hsv_to_rgb()的rgb颜色的取值范围在【0,1】，而RBG取值范围在【0,255】，所以乘上255
        random.seed(10101)  # Fixed seed for consistent colors across runs.产生随机种子。固定种子为一致的颜色
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.打乱颜色以消除相邻类的关联
        random.seed(None)  # Reset seed to default.重置种子为默认

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))        #K.placeholder:keras中的占位符
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,   #######################################################################question-2：yolo_eval()
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)        #yolo_eval():yolo评估函数
        return boxes, scores, classes

    # def detect_image(self, image):
    #     if self.is_fixed_size:
    #         assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
    #         assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
    #         boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))   #进行缩放
    #         #letterbox_image()：先生成一个用绝对灰（RGB：128,128,128）填充的416*416的新图片，然后按照比例缩放（采样方法：BICBIC）后的输入图片粘贴，粘贴不到的部分保留灰色
    #     else:
    #         # model_image_size定义的宽和高必须是32的整倍数；若没有定义model_image_size，将输入图片的尺寸调整到32的整倍数，并调用letterbox_image()函数进行缩放
    #         new_image_size = (image.width - (image.width % 32),
    #                           image.height - (image.height % 32))
    #         boxed_image = letterbox_image(image, new_image_size)     #进行缩放
    #     image_data = np.array(boxed_image, dtype='float32')
    #
    #     #print(image_data.shape)
    #     image_data /= 255.    #将缩放后图片的数值除以255，做归一化
    #     # 将（416,416,3）数组调整为（1,416,416,3）元组，满足YOLOv3输入的张量格式
    #     image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    #
    #     out_boxes, out_scores, out_classes = self.sess.run(
    #         [self.boxes, self.scores, self.classes],
    #         feed_dict={        #输入参数
    #             self.yolo_model.input: image_data,    #输入图片
    #             self.input_image_shape: [image.size[1], image.size[0]],      #图片尺寸416x416
    #             K.learning_phase(): 0         #学习模式：0测试/1训练
    #         })
    #     return_boxs = []
    #     for i, c in reversed(list(enumerate(out_classes))):
    #         predicted_class = self.class_names[c]    #目标类别的名字
    #         if predicted_class != 'person' :
    #             continue
    #         box = out_boxes[i]        #目标框
    #        # score = out_scores[i]    #目标框的置信度评分
    #         x = int(box[1])
    #         y = int(box[0])
    #         w = int(box[3]-box[1])
    #         h = int(box[2]-box[0])
    #         if x < 0 :
    #             w = w + x
    #             x = 0
    #         if y < 0 :
    #             h = h + y
    #             y = 0
    #         return_boxs.append([x,y,w,h])
    #
    #     return return_boxs
    def detect_image(self, image):
        if self.is_fixed_size:
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else: 
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.   #归一化
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,                       ##################################################question：yolo_model.input:image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0 #K.learning_phase() 学习阶段标志（0=test，1=train），它作为输入传递给任何的Keras函数
            })
        return_boxs = []
        return_class_name = []
        person_counter = 0
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            # print(self.class_names[c])
            '''
            if predicted_class != 'person' and predicted_class != 'car':
               print(predicted_class)
               continue
            '''
            if predicted_class != args["class"]:
                # print(predicted_class)
                continue

            person_counter += 1
            # if  predicted_class != 'car':
            # continue
            # label = predicted_class
            box = out_boxes[i]
            # score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxs.append([x, y, w, h])
            # print(return_boxs)
            return_class_name.append([predicted_class])
        # cv2.putText(image, str(self.class_names[c]),(int(box[0]), int(box[1] -50)),0, 5e-3 * 150, (0,255,0),2)
        # print("Found person: ",person_counter)
        return return_boxs, return_class_name

    def close_session(self):
        self.sess.close()
