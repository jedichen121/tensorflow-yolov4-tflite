# Copyright (c) 2021 Jiyang Chen
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import cv2
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, YOLOv4_tiny, decode
from PIL import Image
from core.config import cfg

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', './data/yolov4.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')

def main(_argv):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
        XYSCALE = cfg.YOLO.XYSCALE
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    input_size = FLAGS.size
    image_path = FLAGS.image

    image_folder = "./data/aerial_photos/"
    # Find all png files within the given directory, sorted numerically
    image_files = []
    file_names = os.listdir(image_folder)

    for file in file_names:
        if ".JPG" in file:
            # image_files.append(os.path.join(image_folder, file))
            image_files.append(file)
    image_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    if FLAGS.tiny:
        if FLAGS.model == 'yolov3':
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        else:
            feature_maps = YOLOv4_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode(fm, NUM_CLASS, i)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
        # model.summary()
        utils.load_weights_tiny(model, FLAGS.weights, FLAGS.model)
    else:
        if FLAGS.model == 'yolov3':
            feature_maps = YOLOv3(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_v3(model, FLAGS.weights)
        elif FLAGS.model == 'yolov4':
            feature_maps = YOLOv4(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)

            if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                utils.load_weights(model, FLAGS.weights)
            else:
                model.load_weights(FLAGS.weights).expect_partial()


    for image in image_files[:2]:
        original_image = cv2.imread(os.path.join(image_folder, image))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_size = original_image.shape[:2]

        image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        partial_image = original_image[int(original_image_size[0]/3):int(original_image_size[0]/3*2), 
                                       int(original_image_size[1]/3):int(original_image_size[1]/3*2), :]
        # partial_image = original_image[int(original_image_size[0]/5*2):int(original_image_size[0]/5*3), 
        #                                int(original_image_size[1]/5*2):int(original_image_size[1]/5*3), :]
        partial_image_size = partial_image.shape[:2]
        image_data = utils.image_preprocess(np.copy(partial_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
    
        pred_bbox = model.predict_on_batch(image_data)

        if FLAGS.model == 'yolov4':
            if FLAGS.tiny:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
            else:
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        else:
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
        bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        # bboxes = utils.postprocess_boxes(pred_bbox, partial_image_size, input_size, 0.25)
        bboxes = utils.nms(bboxes, 0.213, method='nms')

        detect_person = False
        for bbox in bboxes:
            class_id = int(bbox[-1])
            score = bbox[-2]
            if class_id == 0:
                detect_person = True
                print('%s: %.2f' % (image, score))
                break

        if not detect_person:
            print('%s: %.2f' % (image, 0))
        
        image = utils.draw_bbox(original_image, bboxes)
        # image = utils.draw_bbox(partial_image, bboxes)
        # image = Image.fromarray(image)
        # image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(FLAGS.output, image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
