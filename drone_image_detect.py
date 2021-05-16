#! /usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 Jiyang Chen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# File name   : visualize.py
# Author      : Jiyang Chen
# Created date: 2021-02-09
# Description : Test if YOLO can detect a person in the drone picture and report its precision.
#               This is used for the multi-objective project.

import os
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import core.utils as utils
from PIL import Image
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, YOLOv4_tiny, decode
from core.yolov4 import filter_boxes
from core.config import cfg
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_size = FLAGS.size
    image_path = FLAGS.image

    image_folder = "./data/aerial_photos/"
    # Find all png files within the given directory, sorted numerically
    image_files = []
    file_names = os.listdir(image_folder)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    for file in file_names:
        if ".JPG" in file:
            # image_files.append(os.path.join(image_folder, file))
            image_files.append(file)
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


    for image in image_files[:]:
        original_image = cv2.imread(os.path.join(image_folder, image))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]
        image_data = cv2.resize(original_image, (input_size, input_size))

        # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        partial_image = original_image[int(original_image_size[0]/3):int(original_image_size[0]/3*2), 
                                       int(original_image_size[1]/3):int(original_image_size[1]/3*2), :]
        # partial_image = original_image[int(original_image_size[0]/5*2):int(original_image_size[0]/5*3), 
        #                                int(original_image_size[1]/5*2):int(original_image_size[1]/5*3), :]
        image_data = cv2.resize(partial_image, (input_size, input_size))

        image_data = image_data / 255.
        # image_data = image_data[np.newaxis, ...].astype(np.float32)

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)
        
        batch_data = tf.constant(images_data)

        pred_bbox = infer(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        # image = utils.draw_bbox(original_image, pred_bbox)

        detect_person = False
        _, out_scores, out_classes, num_boxes = pred_bbox
        # print("num_boxes is ", num_boxes)
        # print("out_class is ", out_classes)
        for i in range(num_boxes[0]):
            class_id = int(out_classes[0][i])
            # print(class_id)
            if class_id == 0:
                detect_person = True
                print('%s: %.2f' % (image, out_scores[0][i]))
                break

        if not detect_person:
            print('%s: %.2f' % (image, 0))

        # image = utils.draw_bbox(image_data*255, pred_bbox)
        # image = Image.fromarray(image.astype(np.uint8))
        # image.show()
        # image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # cv2.imwrite(FLAGS.output, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
