#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : main.py
#   Author      : YunYang1994
#   Created date: 2020-02-23 16:34:47
#   Description :
#
#================================================================


import os
import cv2
import time
import numpy as np
import tensorflow as tf
from mtcnn import pnet, rnet, onet
from MobileFaceNet import MobileFaceNet
from utils import detect_face, align_face, recognize_face

model = MobileFaceNet()
cv2.namedWindow("detecting face")
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret,image = cap.read()
    if ret == True:
        # resize image
        image_h, image_w, _ = image.shape
        new_h, new_w = int(0.5*image_h), int(0.5*image_w)
        image = cv2.resize(image, (new_w, new_h))

        org_image = image.copy()
        # detecting faces
        # t1 = time.time()
        image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
        total_boxes, points = detect_face(image, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)
        # t2 = time.time()
        # print("time: %.2fms" %((t2-t1)*1000))

        for idx, (bounding_box, keypoints) in enumerate(zip(total_boxes, points.T)):
            bounding_boxes = {
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1],
                    'keypoints': {
                            'left_eye': (int(keypoints[0]), int(keypoints[5])),
                            'right_eye': (int(keypoints[1]), int(keypoints[6])),
                            'nose': (int(keypoints[2]), int(keypoints[7])),
                            'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                            'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                    }
                }

            bounding_box = bounding_boxes['box']
            keypoints = bounding_boxes['keypoints']

            cv2.circle(org_image,(keypoints['left_eye']),   2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['right_eye']),  2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['nose']),       2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['mouth_left']), 2, (255,0,0), 3)
            cv2.circle(org_image,(keypoints['mouth_right']),2, (255,0,0), 3)
            cv2.rectangle(org_image,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,255,0), 2)
            # align face and extract it out
            align_image = align_face(image, keypoints)

            marigin = 16
            xmin = max(bounding_box[0] - marigin, 0)
            ymin = max(bounding_box[1] - marigin, 0)
            xmax = min(bounding_box[0] + bounding_box[2] + marigin, new_w)
            ymax = min(bounding_box[1] + bounding_box[3] + marigin, new_h)

            crop_image = align_image[ymin:ymax, xmin:xmax, :]
            if crop_image is not None:
                t1 = time.time()
                embedding = model(crop_image)
                person = recognize_face(embedding)
                cv2.putText(org_image, person, (bounding_box[0], bounding_box[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                                                                    1., (0, 0, 255), 3, lineType=cv2.LINE_AA)
                t2 = time.time()
                print("time: %.2fms" %((t2-t1)*1000))

        cv2.imshow('detecting face', org_image)
        if cv2.waitKey(30) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
