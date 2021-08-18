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

"""Object detection on camera frames using OpenCV. Streaming results to DELC system

TEST_DATA=../all_models

Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt

"""
import argparse
import collections
import common
import cv2
import os
import re
import numpy as np
from PIL import Image
import csv
import time
import datetime
import tflite_runtime.interpreter as tflite
import glob

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

    top_k = min(top_k, len(scores))
    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

import asyncio

async def saving(pil_im, inference_time, img_name, class_names, scores, boxes, images_dir, predictions_dir):
    # save the detection results
    start_time = time.monotonic()
    if len(class_names) > 0:
        # save the image file
        t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = img_name + ".jpg"
        image_path = os.path.join(images_dir, image_name)
        pil_im.save(image_path)
        
        # save the prediction
        csv_name = img_name + ".csv"
        csv_path = os.path.join(predictions_dir, csv_name)
        f = open(csv_path, 'w')
        with f:
            fnames = ['fps', 'timestamp', 'width', 'height', 'image_name', 'image_path', 'class_names', 'scores', 'boxes']
            writer = csv.DictWriter(f, fieldnames=fnames)
            writer.writeheader()
            (width, height) = pil_im.size
            fps = int(1000./inference_time)
            writer.writerow({'fps': fps, 'timestamp': t, 'width' : width, 'height': height, 'image_name': image_name, 'image_path': image_path, 
                            'class_names': str(class_names), 'scores': str(scores), 'boxes': str(boxes)})
        
    end_time = time.monotonic()
    #print('Saving time: {:.2f} ms'.format((end_time - start_time) * 1000))

async def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=100,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
                        
    parser.add_argument('--input_path', type=str, default='',
                        help='Input path for the testing video')
    parser.add_argument('--output_path', type=str, default='',
                        help='Output path to save the results')
    args = parser.parse_args()
    
    # make output dirs
    detection_dir = os.path.join(args.output_path)
    if not os.path.isdir(detection_dir):
        os.makedirs(detection_dir)
    images_dir = os.path.join(detection_dir, 'images')
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)
    predictions_dir = os.path.join(detection_dir, 'predictions')
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    #cap = cv2.VideoCapture(args.camera_idx)
    #cap = cv2.VideoCapture(args.input_path)

    #while cap.isOpened():
    for f in glob.glob(args.input_path + "*.jpg"):
        #ret, frame = cap.read()
        #if not ret:
        #    break
        #cv2_im = frame
        cv2_im = cv2.imread(f)
        img_name = os.path.basename(f).split(".")[0]

        #start_time = time.monotonic()
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)
        
        start_time = time.monotonic()
        interpreter.invoke()
        end_time = time.monotonic()
        inference_time = (end_time - start_time)*1000
        
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        
        # show the results
        #cv2_im = append_objs_to_img(cv2_im, objs, labels)
        #cv2.imshow('frame', cv2_im)
        #end_time = time.monotonic()
        #print('Showing time: {:.2f} ms'.format((end_time - start_time) * 1000))
        
        # save detection results
        class_names, scores, boxes = get_detection_results(objs, labels)
        await saving(pil_im, inference_time, img_name, class_names, scores, boxes, images_dir, predictions_dir)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cap.release()
    #cv2.destroyAllWindows()

def get_detection_results(objs, labels):
    class_names = []
    scores = []
    boxes = []
    for obj in objs:
        xmin, ymin, xmax, ymax = list(obj.bbox)
        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(obj.score)
        class_names.append(labels.get(obj.id, obj.id))

    return class_names, scores, boxes

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        '''is_human = labels.get(obj.id, obj.id)
        if (is_human != 'person'):
            continue'''

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    asyncio.run(main())
