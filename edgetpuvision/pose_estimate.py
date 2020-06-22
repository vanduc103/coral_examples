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

"""A demo which runs object detection on camera frames.

export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data

Run face detection model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite

Run coco model:
python3 -m edgetpuvision.detect \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""

import argparse
import collections
import colorsys
import itertools
import time

from . import svg
from . import utils
from .apps import run_app

from .pose_engine import PoseEngine

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

TRAINING = (
    #('left shoulder', 'left wrist'),
    #('right shoulder', 'right wrist'),
    #('left wrist', 'left hip'),
    #('right wrist', 'right hip'),
    ('left hip', 'left ankle'),
    #('right hip', 'right ankle'),
)
TRAINING_SIZE = 4
training = [None] * TRAINING_SIZE

CSS_STYLES = str(svg.CssStyle({'.back': svg.Style(fill='black',
                                                  stroke='black',
                                                  stroke_width='0.5em'),
                               '.bbox': svg.Style(fill_opacity=0.0,
                                                  stroke_width='0.1em')}))

BBox = collections.namedtuple('BBox', ('x', 'y', 'w', 'h'))
BBox.area = lambda self: self.w * self.h
BBox.scale = lambda self, sx, sy: BBox(x=self.x * sx, y=self.y * sy,
                                       w=self.w * sx, h=self.h * sy)
BBox.__str__ = lambda self: 'BBox(x=%.2f y=%.2f w=%.2f h=%.2f)' % self

Object = collections.namedtuple('Object', ('id', 'label', 'score', 'bbox'))
Object.__str__ = lambda self: 'Object(id=%d, label=%s, score=%.2f, %s)' % self

def size_em(length):
    return '%sem' % str(0.6 * (length + 1))

def color(i, total):
    return tuple(int(255.0 * c) for c in colorsys.hsv_to_rgb(i / total, 1.0, 1.0))

def make_palette(keys):
    return {key : svg.rgb(color(i, len(keys))) for i, key in enumerate(keys)}

def make_get_color(color, labels):
    if color:
        return lambda obj_id: color

    if labels:
        palette = make_palette(labels.keys())
        return lambda obj_id: palette[obj_id]

    return lambda obj_id: 'white'

def caldist(x1, y1, x2, y2):
    import math
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def decretest(A):
    for i in range( len(A) - 1 ):
        if A[i] <= A[i+1]:
            return False
    return True

def incretest(A):
    for i in range( len(A) - 1 ):
        if A[i] >= A[i+1]:
            return False
    return True

def overlay(engine, title, objs, inference_size, inference_time, layout, idx, upcnt, downcnt, threshold=0.2):
    x0, y0, width, height = layout.window
    font_size = 0.03 * height

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=width, height=height,
                  viewBox='%s %s %s %s' % layout.window,
                  font_size=font_size, font_family='monospace', font_weight=500)
    doc += defs

    box_x, box_y, box_w, box_h = 0, 0, inference_size[0], inference_size[1]
    scale_x, scale_y = width / box_w, height / box_h
    for pose in objs:
        xys = {}
        kp_dist = {} # distance between keypoints
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < threshold: continue
            percent = int(100 * keypoint.score)

            # Offset and scale to source coordinate space.
            kp_y = int((keypoint.yx[0] - box_y) * scale_y)
            kp_x = int((keypoint.yx[1] - box_x) * scale_x)
            xys[label] = (kp_x, kp_y)

            doc += svg.Circle(cx=kp_x, cy=kp_y, r=5, fill='cyan')

        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            ax, ay = xys[a]
            bx, by = xys[b]
            doc += svg.Line(x1=ax, y1=ay, x2=bx, y2=by, stroke='black', stroke_width=1)

        for a, b in TRAINING:
            if a not in xys or b not in xys: continue
            ax, ay = xys[a]
            bx, by = xys[b]
            dist = caldist(ax, ay, bx, by)
            training[idx % TRAINING_SIZE] = int(dist)
            idx += 1

    ox = x0 + 20
    oy1, oy2 = y0 + 20 + font_size, y0 + height - 20

    # Title
    if title:
        doc += svg.Rect(x=0, y=0, width=size_em(len(title)), height='1em',
                        transform='translate(%s, %s) scale(1,-1)' % (ox, oy1), _class='back')
        doc += svg.Text(title, x=ox, y=oy1, fill='white')

    # Info
    '''lines = [
        'Objects: %d' % len(objs),
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
    ]'''

    # Training Info
    lines = []
    if not (None in training):
        #print(training)
        label1 = 'Standing UP (' + str(upcnt) + ')'
        label2 = 'Sitting DOWN (' + str(downcnt) + ')'
        if incretest(training):
            upcnt += 1
            label1 = 'Standing UP (' + str(upcnt) + ')'
        elif decretest(training):
            downcnt += 1
            label2 = 'Sitting DOWN (' + str(downcnt) + ')'
        lines = [
            'Workout: %s, %s' % (label1, label2),
            'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
        ]

    for i, line in enumerate(reversed(lines)):
        y = oy2 - i * 1.7 * font_size
        doc += svg.Rect(x=0, y=0, width=size_em(len(line)), height='1em',
                       transform='translate(%s, %s) scale(1,-1)' % (ox, y), _class='back')
        doc += svg.Text(line, x=ox, y=y, fill='white')

    return str(doc), idx, upcnt, downcnt


def convert(obj, labels):
    x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
    return Object(id=obj.label_id,
                  label=labels[obj.label_id] if labels else None,
                  score=obj.score,
                  bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0))

def print_results(objs):
    from datetime import datetime
    print(datetime.now())
    #print('\nInference (rate=%.2f fps):' % inference_rate)
    for i, obj in enumerate(objs):
        print('    %d: %s, area=%.2f' % (i, obj, obj.bbox.area()))


from PIL import Image
import numpy as np
from datetime import datetime
#import cv2
def render_gen(args):
    fps_counter  = utils.avg_fps_counter(30)

    engines, titles = utils.make_engines(args.model, PoseEngine)
    #assert utils.same_input_image_sizes(engines)
    engines = itertools.cycle(engines)
    engine = next(engines)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])

    draw_overlay = True

    yield utils.input_image_size(engine)

    output = None
    idx = 0
    upcnt, downcnt = 0, 0
    while True:
        tensor, layout, command = (yield output)

        inference_rate = next(fps_counter)
        if draw_overlay:
            start = time.monotonic()
            inf_output = engine.run_inference(tensor)
            outputs, _ = engine.ParseOutput(inf_output)
            inference_time = time.monotonic() - start

            if args.print and len(outputs) > 0:
                print_results(outputs)

            title = titles[engine]
            output, idx, upcnt, downcnt = overlay(engine, title, outputs, inference_size, inference_time, layout, idx, upcnt, downcnt)
        else:
            output = None

        if command == 'o':
            draw_overlay = not draw_overlay
        elif command == 'n':
            engine = next(engines)


def add_render_gen_args(parser):
    parser.add_argument('--model',
                        help='.tflite model path', required=True)
    parser.add_argument('--labels',
                        help='labels file path')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Max number of objects to detect')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--min_area', type=float, default=0.0,
                        help='Min bounding box area')
    parser.add_argument('--max_area', type=float, default=1.0,
                        help='Max bounding box area')
    parser.add_argument('--filter', default=None,
                        help='Comma-separated list of allowed labels')
    parser.add_argument('--color', default=None,
                        help='Bounding box display color'),
    parser.add_argument('--print', default=False, action='store_true',
                        help='Print inference results')
    parser.add_argument('--save', default=False, action='store_true',
                        help='Save detected objects')

def main():
    run_app(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
