#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np   #numpy针对数组运算提供大量的数学函数库
from PIL import Image  #PIL第三方图像处理库
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3   #余弦距离度量的门限值(对象外在表现)
    nn_budget = None        #外观描述符库的最大大小
    nms_max_overlap = 1.0      #最大检测重叠(非最大抑制阈值)
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture('./2.mp4')   #参数0表示打开笔记本的内置镜头

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))            #获取视频流的帧的宽度
        h = int(video_capture.get(4))            #获取视频流的帧的宽度
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')          #进行视频编码的格式，也可以使用cv2.VideoWriter_fourcc('M','J','P','G')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')          #创建并写数据入detection.txt文件中
        frame_index = -1   #用于计数，索引（0123456....）
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  #  read()按帧读取视频  ret为bool   frame  三维矩阵：640*480*3
        if ret != True:
            break
        t1 = time.time()             #返回当前时间的时间戳

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names= yolo.detect_image(image)   #图像边缘检测
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.非极大值抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker返回追踪路径
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)


        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            
        cv2.imshow('', frame)   #显示图片
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)   #打印
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
