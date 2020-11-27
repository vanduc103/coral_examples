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

import argparse
import collections
import cv2
import numpy as np
import os
import csv
import glob
import re

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

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # output dict
    detect_output, face_output, pose_output = [], [], []

    # csv reader for detect
    csvfile = open('detect_output.csv', 'r')
    with csvfile:
        fnames = ['timestamp', 'idx', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
        reader = csv.DictReader(csvfile)
        for row in reader:
            obj = {}
            obj['timestamp'] = row['timestamp']
            obj['label'] = row['label']
            obj['idx'] = row['idx']
            obj['width'] = row['width']
            obj['height'] = row['height']
            obj['xmin'] = row['xmin']
            obj['ymin'] = row['ymin']
            obj['xmax'] = row['xmax']
            obj['ymax'] = row['ymax']
            obj['score'] = row['score']
            detect_output.append(obj)

    # csv reader for face
    if os.path.isfile('face_output.csv'):
        csvfile = open('face_output.csv', 'r')
        with csvfile:
            fnames = ['timestamp', 'idx', 'label', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'score']
            reader = csv.DictReader(csvfile)
            for row in reader:
                obj = {}
                obj['timestamp'] = row['timestamp']
                obj['label'] = row['label']
                obj['idx'] = row['idx']
                obj['width'] = row['width']
                obj['height'] = row['height']
                obj['xmin'] = row['xmin']
                obj['ymin'] = row['ymin']
                obj['xmax'] = row['xmax']
                obj['ymax'] = row['ymax']
                obj['score'] = row['score']
                face_output.append(obj)

    # csv reader for pose
    if os.path.isfile('pose_output.csv'):
        csvfile = open('pose_output.csv', 'r')
        with csvfile:
            fnames = ['timestamp', 'idx', 'label', 'width', 'height', 'x', 'y', 'score']
            reader = csv.DictReader(csvfile)
            for row in reader:
                obj = {}
                obj['timestamp'] = row['timestamp']
                obj['label'] = row['label']
                obj['idx'] = row['idx']
                obj['width'] = row['width']
                obj['height'] = row['height']
                obj['x'] = row['x']
                obj['y'] = row['y']
                obj['score'] = row['score']
                pose_output.append(obj)

    # read frames
    for image_path in sorted(glob.glob("/home/duclv/homework/dataset/katech/성선영/도심로 2차/20191018_162514(방이 삼거리 구간 좌회전 시나리오)_141f/3/*.jpg")):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        #print(image_name)
        cv2_im = cv2.imread(image_path)
        w, h, _ = cv2_im.shape
        #cv2_im = np.zeros((w, h, 3), dtype=np.uint8) + 160

        objs = []
        for obj in detect_output:
            if obj['timestamp'] == image_name:
                objs.append(obj)
        if len(objs) > 0:
            cv2_im = append_objs_to_img(cv2_im, objs)

        face_objs = []
        for obj in face_output:
            if obj['timestamp'] == image_name:
                face_objs.append(obj)
        if len(face_objs) > 0:
            cv2_im = append_objs_to_img(cv2_im, face_objs)

        poses = []
        idx = -1
        pose = []
        for obj in pose_output:
            if obj['timestamp'] == image_name:
                if int(obj['idx']) > idx:
                    if len(pose) > 0:
                        poses.append(pose)
                    pose = []
                    idx += 1
                pose.append(obj)
        if len(poses) > 0:
            cv2_im = draw_skel_and_kp(cv2_im, poses)

        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            continue

    # create video from images
    '''out = cv2.VideoWriter('out.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()'''

def append_objs_to_img(cv2_im, objs, color=(0, 255, 0), min_detect_score=0.1):
    for obj in objs:
        if float(obj['score']) < min_detect_score: continue
        x0, y0, x1, y1 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
        percent = int(100 * float(obj['score']))
        label = '{}% {}'.format(percent, obj['label'])

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), color, 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

def draw_skel_and_kp(
        img, poses, 
        min_pose_score=0.3, min_part_score=0.2):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []

    for pose in poses:
        xys = {}
        points = [(-1., -1.)] * 17
        for keypoint in pose:
            if float(keypoint['score']) < min_part_score: continue
            # Coord
            kp_y = float(keypoint['y'])
            kp_x = float(keypoint['x'])
            label = keypoint['label']
            xys[label] = (kp_x, kp_y)
            cv_keypoints.append(cv2.KeyPoint(int(kp_x), int(kp_y), 10. * float(keypoint['score'])))
            points[BODY_PARTS[label]] = (int(kp_x), int(kp_y))

        results = []
        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            ax, ay = xys[a]
            bx, by = xys[b]
            results.append(np.array([[ax, ay], [bx, by]]).astype(np.int32),)
        adjacent_keypoints.extend(results)

    
    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(0, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(0, 255, 255), thickness=2)
    return out_img

if __name__ == '__main__':
    main()
