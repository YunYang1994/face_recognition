#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : register_face.py
#   Author      : YunYang1994
#   Created date: 2020-02-23 22:09:11
#   Description :
#
#================================================================


import os
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf
from mtcnn import pnet, rnet, onet
from MobileFaceNet import MobileFaceNet
from utils import detect_face, align_face


def extract_oneface(image, marigin=16):
    # detecting faces
    image = cv2.cvtColor(image ,cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    total_boxes, points = detect_face(image, 20, pnet, rnet, onet, [0.6, 0.7, 0.7], 0.709)
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

        # align face and extract it out
        align_image = align_face(image, keypoints)
        align_image = cv2.cvtColor(align_image ,cv2.COLOR_RGB2BGR)

        xmin = max(bounding_box[0] - marigin, 0)
        ymin = max(bounding_box[1] - marigin, 0)
        xmax = min(bounding_box[0] + bounding_box[2] + marigin, w)
        ymax = min(bounding_box[1] + bounding_box[3] + marigin, h)

        crop_image = align_image[ymin:ymax, xmin:xmax, :]
        # "just need only one face"
        return crop_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-person", type=str)
    parser.add_argument('-camera', action='store_true', default=False)
    args = parser.parse_args()

    model = MobileFaceNet()
    person_path = "./database/%s" %(args.person)
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    if args.camera:
        img_idx = 0

        cv2.namedWindow("detecting face")
        cap = cv2.VideoCapture(0)

        while(cap.isOpened()):
            ret,image = cap.read()
            if ret == True:
                # resize image
                image_h, image_w, _ = image.shape
                new_h, new_w = int(0.5*image_h), int(0.5*image_w)
                image = cv2.resize(image, (new_w, new_h))
                face = extract_oneface(image)
                if face is None: continue

                h, w, _ = face.shape
                image[:h, :w, :] = face
                cv2.putText(image, "%d/10" %img_idx, (new_w-100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                cv2.imshow('detecting face', image)
                key = cv2.waitKey(30)
                if key == ord('q') or img_idx == 10:
                    break
                elif key == ord('s'):
                    cv2.imwrite("./database/%s/%d.jpg" %(args.person, img_idx), face)
                    img_idx += 1

        cap.release()
        cv2.destroyAllWindows()

    else:
        for img_path in glob.glob(person_path+"/*.jpg"):
            print(img_path)
            image = cv2.imread(img_path)
            face = extract_oneface(image)
            cv2.imwrite(img_path, face)

    image_path = "./database/%s" %(args.person)
    image_list = glob.glob(image_path + "/*.jpg")

    embeddings = []
    for im_path in image_list:
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embeddings.append(model(image))

    embedding = np.concatenate(embeddings, 0).mean(0).flatten()
    np.save("./database/%s/%s" %(args.person, args.person), embedding)

