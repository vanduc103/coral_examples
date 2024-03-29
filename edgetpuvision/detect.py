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

from edgetpu.detection.engine import DetectionEngine

from . import svg
from . import utils
from .apps import run_app

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

def overlay(title, objs, get_color, inference_time, inference_rate, layout):
    x0, y0, width, height = layout.window
    font_size = 0.03 * height

    defs = svg.Defs()
    defs += CSS_STYLES

    doc = svg.Svg(width=width, height=height,
                  viewBox='%s %s %s %s' % layout.window,
                  font_size=font_size, font_family='monospace', font_weight=500)
    doc += defs

    for obj in objs:
        percent = int(100 * obj.score)
        if obj.label:
            caption = '%d%% %s' % (percent, obj.label)
        else:
            caption = '%d%%' % percent

        x, y, w, h = obj.bbox.scale(*layout.size)
        color = get_color(obj.id)

        doc += svg.Rect(x=x, y=y, width=w, height=h,
                        style='stroke:%s' % color, _class='bbox')
        doc += svg.Rect(x=x, y=y+h ,
                        width=size_em(len(caption)), height='1.2em', fill=color)
        t = svg.Text(x=x, y=y+h, fill='black')
        t += svg.TSpan(caption, dy='1em')
        doc += t

    ox = x0 + 20
    oy1, oy2 = y0 + 20 + font_size, y0 + height - 20

    # Title
    if title:
        doc += svg.Rect(x=0, y=0, width=size_em(len(title)), height='1em',
                        transform='translate(%s, %s) scale(1,-1)' % (ox, oy1), _class='back')
        doc += svg.Text(title, x=ox, y=oy1, fill='white')

    # Info
    lines = [
        'Objects: %d' % len(objs),
        'Inference time: %.2f ms (%.2f fps)' % (inference_time * 1000, 1.0 / inference_time)
    ]

    for i, line in enumerate(reversed(lines)):
        y = oy2 - i * 1.7 * font_size
        doc += svg.Rect(x=0, y=0, width=size_em(len(line)), height='1em',
                       transform='translate(%s, %s) scale(1,-1)' % (ox, y), _class='back')
        doc += svg.Text(line, x=ox, y=y, fill='white')

    return str(doc)


def convert(obj, labels):
    x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
    return Object(id=obj.label_id,
                  label=labels[obj.label_id] if labels else None,
                  score=obj.score,
                  bbox=BBox(x=x0, y=y0, w=x1 - x0, h=y1 - y0))

def print_results(inference_rate, objs):
    print('\nInference (rate=%.2f fps):' % inference_rate)
    for i, obj in enumerate(objs):
        print('    %d: %s, area=%.2f' % (i, obj, obj.bbox.area()))


from PIL import Image
import numpy as np
from datetime import datetime
import cv2
import math

def render_gen(args):
    fps_counter  = utils.avg_fps_counter(30)

    engines, titles = utils.make_engines(args.model, DetectionEngine)
    assert utils.same_input_image_sizes(engines)
    engines = itertools.cycle(engines)
    engine = next(engines)

    labels = utils.load_labels(args.labels) if args.labels else None
    filtered_labels = set(l.strip() for l in args.filter.split(',')) if args.filter else None
    print(filtered_labels)
    get_color = make_get_color(args.color, labels)

    # hand tracking engine
    if args.hand:
        engines_hand, _ = utils.make_engines('/home/mendel/google-coral/examples-camera/all_models/hand_tflite_graph_edgetpu.tflite', DetectionEngine)
        engines_hand = itertools.cycle(engines_hand)
        engine_hand = next(engines_hand)
        labels_hand = utils.load_labels('/home/mendel/google-coral/examples-camera/all_models/hand_label.txt')
    # face detection engine
    if args.face:
        engines_face, _ = utils.make_engines('/home/mendel/google-coral/examples-camera/all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite', DetectionEngine)
        engines_face = itertools.cycle(engines_face)
        engine_face = next(engines_face)

    draw_overlay = True

    yield utils.input_image_size(engine)

    # imagezmq sender
    if args.zmq:
        import socket
        import imagezmq
        from datetime import datetime
        sender = imagezmq.ImageSender(connect_to='tcp://*:5556', REQ_REP=False) # REQ_REP=False: use PUB/SUB (non-block)
        #rpi_name = socket.gethostname()

    output = None
    while True:
        tensor, layout, command = (yield output)

        inference_rate = next(fps_counter)
        if draw_overlay:
            start = time.monotonic()
            objs = engine .detect_with_input_tensor(tensor, threshold=0.5, top_k=10)
            im = tensor
            # imagezmq send image
            if args.zmq:
                W, H = utils.input_image_size(engine)
                im = np.reshape(im, (W, H, 3))
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                timestamp = datetime.timestamp(datetime.now())
                sender.send_image(timestamp, im)

            if args.hand:
                objs_hand = engine_hand .detect_with_input_tensor(tensor, threshold=0.1, top_k=1)
            if args.face:
                W, H = utils.input_image_size(engine)
                im = np.reshape(im, (W, H, 3))
                im = Image.fromarray(im)
                objs_face = engine_face .detect_with_image(im, threshold=0.5, top_k=10)
            inference_time = time.monotonic() - start
            objs = [convert(obj, labels) for obj in objs]
            if args.hand:
                objs_hand = [convert(obj, labels_hand) for obj in objs_hand]
            if args.face:
                objs_face = [convert(obj, None) for obj in objs_face]

            if labels and filtered_labels:
                objs = [obj for obj in objs if obj.label in filtered_labels]

            objs = [obj for obj in objs if args.min_area <= obj.bbox.area() <= args.max_area]
            if args.hand:
                objs = objs + objs_hand
            if args.face:
                objs = objs + objs_face

            if args.save and len(objs) > 0:
                W, H = utils.input_image_size(engine)
                im = np.reshape(im, (W, H, 3))
                im = Image.fromarray(im)
                for obj in objs:
                    x, y, w, h = obj.bbox
                    x, y, w, h = int(x * W), int(y * H), int(w * W), int(h * H) # scale up to real size
                    crop_rectangle = (x, y, x+w, y+h)
                    det = im.crop(crop_rectangle)
                    det = det.resize((64, 128))
                    #dt_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]
                    dt_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
                    name = obj.label + "-" + dt_str + ".jpg"
                    det.save("images/" + name)

            if args.print and len(objs) > 0:
                print_results(inference_rate, objs)

            title = titles[engine]
            output = overlay(title, objs, get_color, inference_time, inference_rate, layout)
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
    parser.add_argument('--top_k', type=int, default=50,
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
    parser.add_argument('--hand', default=False, action='store_true',
                        help='Use handtracking')
    parser.add_argument('--face', default=False, action='store_true',
                        help='Use Face detection')
    parser.add_argument('--save', default=False, action='store_true',
                        help='Save detected objects')
    parser.add_argument('--zmq', default=False, action='store_true',
                        help='Send frames via ZeroMQ')

def main():
    run_app(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()
