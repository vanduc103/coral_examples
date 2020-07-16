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

import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import cv2
import argparse
import common

from edgetpu.detection.engine import DetectionEngine

BODY_PARTS = {"nose": 0, "left eye": 1, "right eye": 2, "left ear": 3, "right ear": 4,
                      "left shoulder": 5, "right shoulder": 6, "left elbow": 7, "right elbow": 8, "left wrist": 9,
                      "right wrist": 10, "left hip": 11, "right hip": 12, "left knee": 13, "right knee": 14,
                      "left ankle": 15, "right ankle": 16}

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

import zmq
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--model', type=str, help='Pose model to use. ', default = '')
    parser.add_argument('--detect', action='store_true', help='Detect person', default = False)
    parser.add_argument('--filtered_labels', type=str, help='Filtered labels. ', default = '0')
    parser.add_argument('--zmq', action='store_true', help='Send via ZeroMQ', default = False)
    args = parser.parse_args()

    #engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    engine = PoseEngine(args.model)
    _, image_height, image_width, _ = engine.get_input_tensor_shape()

    if args.detect:    
        detect_engine = DetectionEngine('../examples-camera/all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    print("Load all models done!")

    if args.zmq:
        # imagezmq sender
        #import imagezmq
        #sender_img = imagezmq.ImageSender(connect_to='tcp://*:5555', REQ_REP=False) # REQ_REP=False: use PUB/SUB (non-block)
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind("tcp://*:5555")

    cap = cv2.VideoCapture(args.camera_idx)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_im_rgb)
        pil_image.resize((image_width, image_height), Image.NEAREST)

        detect_objs = []
        if args.detect:
            detect_objs = detect_engine.detect_with_image(pil_image,
                                      threshold=0.5,
                                      keep_aspect_ratio=True,
                                      relative_coord=True,
                                      top_k=10)
            if args.filtered_labels:
                detect_objs = [obj for obj in detect_objs if str(obj.label_id) in args.filtered_labels]

        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
        cv2_im, all_points = draw_skel_and_kp(cv2_im, poses, detect_objs)
        #print(all_points.shape)
        
        if args.zmq:
            # imagezmq send image
            #from datetime import datetime
            #timestamp = datetime.timestamp(datetime.now())
            #sender_img.send_image(timestamp, cv2_im_rgb)

            # zmq send points
            timestamp = datetime.timestamp(datetime.now())
            send_array(socket, np.array(all_points).astype(np.float), timestamp)

        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_skel_and_kp(
        img, poses, detect_objs,
        min_pose_score=0.3, min_part_score=0.2):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    all_points = []

    for pose in poses:
        if pose.score < min_pose_score: continue
        xys = {}
        points = [(-1., -1.)] * 17
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < min_part_score: continue
            # Coord
            kp_y = keypoint.yx[0]
            kp_x = keypoint.yx[1]
            xys[label] = (kp_x, kp_y)
            cv_keypoints.append(cv2.KeyPoint(int(kp_x), int(kp_y), 10. * keypoint.score))
            points[BODY_PARTS[label]] = (int(kp_x), int(kp_y))

        all_points.append(np.array(np.stack([p for p in points], axis=0)))

        results = []
        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            ax, ay = xys[a]
            bx, by = xys[b]
            results.append(np.array([[ax, ay], [bx, by]]).astype(np.int32),)
        adjacent_keypoints.extend(results)
    if len(all_points) > 0:
        all_points = np.stack([points for points in all_points], axis=0)

    height, width, channels = img.shape
    for obj in detect_objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        out_img = cv2.rectangle(out_img, (x0, y0), (x1, y1), (0, 255, 0), 1) # fill color

    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(0, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(0, 255, 255), thickness=2)
    return out_img, np.array(all_points)

def send_array(socket, A, msg='None', flags=0, copy=True, track=False):
    """send a numpy array with metadata"""
    md = dict(
        msg = msg,
        dtype = str(A.dtype),
        shape = A.shape,
    )
    socket.send_json(md, flags|zmq.SNDMORE)
    return socket.send(A, flags, copy=copy, track=track)

if __name__ == '__main__':
    main()
