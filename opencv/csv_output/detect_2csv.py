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
import csv
import glob
import time
from PIL import Image
import re
import tflite_runtime.interpreter as tflite

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

def get_output(interpreter, score_threshold, top_k, class_list, image_scale=1.0):
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

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold and class_ids[i] in class_list]

def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='classifier score threshold')
    parser.add_argument('--class_ids', nargs='*', type=int, default=0,
                        help='Array of class id')
    parser.add_argument('--input_files', default='/home/mendel/dataset/*.jpg',
                        help='Input files')
    parser.add_argument('--csv_out', default='detect_output.csv',
                        help='csv output file')
    args = parser.parse_args()
    if args.class_ids == 0:
        args.class_ids = [0]

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    # csv writer
    f = open(args.csv_out, 'w')
    with f:
        fnames = ['timestamp', 'idx', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

        # read frames
        inference_time = []
        for image_path in sorted(glob.glob(args.input_files)):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            #print(image_name)
            pil_im = Image.open(image_path)

            # inference
            start = time.time()
            common.set_input(interpreter, pil_im)
            interpreter.invoke()
            objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k, class_list=args.class_ids)
            inference_time.append(time.time() - start)

            # return results
            (width, height) = pil_im.size
            idx = -1
            for obj in objs:
                x0, y0, x1, y1 = list(obj.bbox)
                x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
                score = obj.score
                label = labels.get(obj.id, obj.id)
                idx += 1
                writer.writerow({'timestamp' : image_name, 'idx': idx, 'label': label, 'width': width, 'height': height, 'xmin': x0, 'ymin': y0, 'xmax': x1, 'ymax': y1, 'score': score})
        
        print("Inference time : {:.3f} ms".format(sum(inference_time)*1000/len(inference_time)))
        print("Frames per second : {:.2f} fps".format(len(inference_time)/sum(inference_time)))
        

if __name__ == '__main__':
    main()
