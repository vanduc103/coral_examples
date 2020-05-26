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

"""A demo that runs hand tracking and object detection on camera frames using OpenCV. 2 EDGETPU
"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
from edgetpu.detection.engine import DetectionEngine
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
    default_model = 'hand_tflite_graph_edgetpu.tflite'
    default_labels = 'hand_label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading Handtracking model {} with {} labels.'.format(args.model, args.labels))
    #interpreter = common.make_interpreter(args.model)
    #interpreter.allocate_tensors()
    engine = DetectionEngine(args.model)
    labels = load_labels(args.labels)

    # for detection
    print('Loading Detection model {} with {} labels.'.format('../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite', '../all_models/coco_labels.txt'))
    #interpreter2 = common.make_interpreter('../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    #interpreter2.allocate_tensors()
    engine2 = DetectionEngine('../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    labels2 = load_labels('../all_models/coco_labels.txt')

    cap = cv2.VideoCapture(args.camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        #common.set_input(interpreter, pil_im)
        #interpreter.invoke()
        #objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        objs = engine.detect_with_image(pil_im,
                                  threshold=0.3,
                                  keep_aspect_ratio=True,
                                  relative_coord=True,
                                  top_k=1)
        cv2_im = append_objs_to_img(cv2_im, objs, labels)

        # detection
        #common.set_input(interpreter2, pil_im)
        #interpreter2.invoke()
        #objs = get_output(interpreter2, score_threshold=0.2, top_k=3)
        objs = engine2.detect_with_image(pil_im,
                                  threshold=0.2,
                                  keep_aspect_ratio=True,
                                  relative_coord=True,
                                  top_k=3)
        cv2_im = append_objs_to_img(cv2_im, objs, labels2)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist() #list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.label_id, obj.label_id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
