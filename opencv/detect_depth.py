# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs object detection on camera frames using OpenCV.

TEST_DATA=../all_models

Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import time
import tflite_runtime.interpreter as tflite
import pyrealsense2 as rs
from matplotlib import pyplot as plt

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color = np.asanyarray(color_frame.get_data())
            colorizer = rs.colorizer()
            colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
            # Create alignment primitive with color as its target stream:
            align = rs.align(rs.stream.color)
            frames = align.process(frames)

            # Update color and depth frames:
            aligned_depth_frame = frames.get_depth_frame()
            colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

            cv2_im = color

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im_rgb)

            start = time.monotonic()
            common.set_input(interpreter, pil_im)
            interpreter.invoke()
            objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
            inference_time = time.monotonic() - start
            inference_time = 'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)

            color = append_objs_to_img(color, objs, labels, inference_time)
            # Calculate the distance
            depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            color = calculate_distance(color, aligned_depth_frame, objs, labels, depth_scale)

            # Stack both images horizontally
            #images = np.hstack((color, colorized_depth))
            #cv2.imwrite("image.jpg", images)

            cv2.imshow('frame', color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()

def append_objs_to_img(cv2_im, objs, labels, inference_time, calculate_distance=False):
    height, width, channels = cv2_im.shape
    #cv2_im = cv2.putText(cv2_im, inference_time, (0, height-30),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    return cv2_im

def calculate_distance(cv2_im, aligned_depth_frame, objs, labels, depth_scale):
    height, width, channels = cv2_im.shape
    i = 0
    for obj in objs:
        i += 1
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = labels.get(obj.id, obj.id)

        depth = np.asanyarray(aligned_depth_frame.get_data())
        # Crop depth data:
        depth = depth[x0:x1,y0:y1].astype(float)

        # Get data scale from the device and convert to meters
        depth = depth * depth_scale
        dist,_,_,_ = cv2.mean(depth)
        distance = "Detected a {0} {1:.3} meters away.".format(label, dist)
        cv2_im = cv2.putText(cv2_im, distance, (0, height-i*30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
    return cv2_im

if __name__ == '__main__':
    main()
