#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : detect_image.py
#   Author      : YunYang1994
#   Created date: 2020-03-19 14:05:53
#   Description :
#
#================================================================


import os
import cv2
import time
import numpy as np
import tensorflow as tf

from PIL import Image, ImageFont, ImageDraw
from mtcnn import pnet, rnet, onet
from models import IResnet
from utils import detect_face, align_face, recognize_face

model = IResnet(tflite_model="IResnet.tflite")
font = ImageFont.truetype('weghts/HuaWenXinWei-1.ttf', 30)
image = cv2.imread("/Users/yangyun/多人照片/5.jpg")

image_h, image_w, _ = image.shape

org_image = image.copy()
image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
total_boxes, points = detect_face(image, 20, pnet, rnet, onet, [0.6, 0.7, 0.9], 0.709)

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
    xmax = min(bounding_box[0] + bounding_box[2] + marigin, image_w)
    ymax = min(bounding_box[1] + bounding_box[3] + marigin, image_h)

    crop_image = align_image[ymin:ymax, xmin:xmax, :]
    if crop_image is not None:
        t1 = time.time()
        embedding = model(crop_image)
        person = recognize_face(embedding)

        org_image_pil = Image.fromarray(org_image)
        draw = ImageDraw.Draw(org_image_pil)
        text_size = draw.textsize(person, font)
        draw.text((bounding_box[0], bounding_box[1]-16), person,  fill=(0, 0, 255), font=font)
        org_image = np.array(org_image_pil)

        t2 = time.time()
        print("time: %.2fms" %((t2-t1)*1000))

org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
image = Image.fromarray(org_image)
image.show()
# image.save("test.png")
