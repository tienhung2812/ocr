# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

sys.path.append(os.getcwd())
from text_detection.nets import model_train as model
from text_detection.utils.rpn_msr.proposal_layer import proposal_layer
from text_detection.utils.text_connector.detectors import TextDetector

from image.transform import *
import calendar;
import time;

tf.app.flags.DEFINE_string('output_path', 'media/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'text_detection/checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS

class TextDetection:
    def __init__(self,image_path):
        self.image_path = image_path
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)


    def resize_image(self,img):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        new_h = int(img_size[0] * im_scale)
        new_w = int(img_size[1] * im_scale)

        new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
        new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

        re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return re_im, (new_h / img_size[0], new_w / img_size[1])

    def get_global_step(self):
        with tf.variable_scope("get_global_step", reuse=tf.AUTO_REUSE):
            v = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        return v


    def find(self):
        if os.path.exists(FLAGS.output_path):
            shutil.rmtree(FLAGS.output_path)
        os.makedirs(FLAGS.output_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        with tf.get_default_graph().as_default():
            input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

            # tf.get_variable_scope().reuse_variables()
            # with tf.variable_scope("icnv1", reuse=tf.AUTO_REUSE):
            # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            global_step = self.global_step
            print("GLOBAL STEP")
            print(global_step)
            bbox_pred, cls_pred, cls_prob = model.model(input_image)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
                model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)


                im_fn =self.image_path
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    return None

                img, (rh, rw) = self.resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                    feed_dict={input_image: [img],
                                                                input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                cropped_image_file_list = []
                for i, box in enumerate(boxes):
                    
                    ts = calendar.timegm(time.gmtime())

                    filename ='/code/media/'+'croped_'+str(i)+'_'+str(ts)+'.png'

                    image = four_point_transform(img, box[:8].reshape(4, 2))
                    cv2.imwrite(filename, image)
                    replace_filename= filename.replace("/code", "")
                    cropped_image_file_list.append(replace_filename)
                    
                for i, box in enumerate(boxes):
                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                thickness=2)

                img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)

                image_result_path = os.path.join(FLAGS.output_path, os.path.basename(im_fn))
                box_result_path = os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt"
                cv2.imwrite(image_result_path, img[:, :, ::-1])
                print(box_result_path)
                with open(box_result_path, "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)
                return image_result_path,  box_result_path, cropped_image_file_list