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
import pose_util

KEYPOINTS = {"nose": 0, "left eye": 1, "right eye": 2, "left ear": 3, "right ear": 4,
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

def to36M(bones, body_parts):
    H36M_JOINTS_17 = [
        'Hip',
        'RHip',
        'RKnee',
        'RFoot',
        'LHip',
        'LKnee',
        'LFoot',
        'Spine',
        'Thorax',
        'Neck/Nose',
        'Head',
        'LShoulder',
        'LElbow',
        'LWrist',
        'RShoulder',
        'RElbow',
        'RWrist',
    ]

    adjusted_bones = []
    bones_len = len(bones)
    for name in H36M_JOINTS_17:
        if not name in body_parts:
            if name == 'Hip':
                adjusted_bones.append((bones[body_parts['RHip']] + bones[body_parts['LHip']]) / 2)
            elif name == 'RFoot':
                adjusted_bones.append(bones[body_parts['RAnkle']])
            elif name == 'LFoot':
                adjusted_bones.append(bones[body_parts['LAnkle']])
            elif name == 'Spine':
                adjusted_bones.append(
                    (
                            bones[body_parts['RHip']] + bones[body_parts['LHip']]
                            + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                    ) / 4
                )
            elif name == 'Thorax':
                adjusted_bones.append(
                    (
                            + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                    ) / 2
                )
            elif name == 'Head':
                thorax = (
                                 + bones[body_parts['RShoulder']] + bones[body_parts['LShoulder']]
                         ) / 2
                adjusted_bones.append(
                    thorax + (
                            bones[body_parts['Nose']] - thorax
                    ) * 2
                )
            elif name == 'Neck/Nose':
                adjusted_bones.append(bones[body_parts['Nose']])
            else:
                raise Exception(name)
        else:
            adjusted_bones.append(bones[body_parts[name]])

    return adjusted_bones

def normalize_2d(pose):
    xs = pose.T[0::2] - pose.T[0]
    ys = pose.T[1::2] - pose.T[1]
    pose = pose.T / np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0)
    mu_x = pose[0].copy()
    mu_y = pose[1].copy()
    pose[0::2] -= mu_x
    pose[1::2] -= mu_y
    return pose.T

def create_pose(model, points):
    common.set_input_pose(model, points)
    model.invoke()
    scores = common.output_tensor(model, 0)

    x = points[:, 0::2]
    y = points[:, 1::2]
    z_pred = np.array(scores)
    z_pred = np.reshape(z_pred, (-1, len(scores)))

    pose = np.stack((x, y, z_pred), axis=-1)
    pose = np.reshape(pose, (len(points), -1))

    return pose


BODY_PARTS = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                      "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9,
                      "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14,
                      "LAnkle": 15, "RAnkle": 16}

POSE_PAIRS = [["Nose", "LEye"], ["Nose", "REye"], ["Nose", "LEar"],
              ["Nose", "REar"], ["LEar", "LEye"], ["REar", "REye"],
              ["LShoulder", "RShoulder"], ["LShoulder", "LElbow"], ["LShoulder", "LHip"], ["RShoulder", "RElbow"],
              ["RShoulder", "RHip"], ["LElbow", "LWrist"], ["RElbow", "RWrist"], ["LHip", "RHip"],
              ["LHip", "LKnee"], ["RHip", "RKnee"], ["LKnee", "LAnkle"], ["RKnee", "RAnkle"]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--model', type=str, help='Pose model to use. ', default = '')
    parser.add_argument('--pose3d', type=str, help='3D Pose model to use. ', default = '')
    parser.add_argument('--dataset', type=str, help='Type of dataset. ', default="CORAL")
    parser.add_argument('--rot', type=int, help='Number of degree to rotate in 3D pose. ', default=90)
    args = parser.parse_args()

    engine = PoseEngine(args.model)
    _, image_height, image_width, _ = engine.get_input_tensor_shape()
    interpreter_3dpose = None
    if len(args.pose3d) > 0:
        interpreter_3dpose = common.make_interpreter(args.pose3d)
        interpreter_3dpose.allocate_tensors()
    print("Load all models done!")

    cap = cv2.VideoCapture(args.camera_idx)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cv2_im_rgb)
        pil_image.resize((image_width, image_height), Image.NEAREST)

        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
        cv2_im = draw_skel_and_kp(cv2_im, poses, args.rot, interpreter_3dpose)
        
        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("frame",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def draw_skel_and_kp(
        img, poses, rot, interpreter_3dpose,
        min_pose_score=0.3, min_part_score=0.2):
    w, h, _ = img.shape
    out_img = np.zeros((w, h, 3), dtype=np.uint8) + 160
    out_img = cv2.line(out_img, (int(h/2), 0), (int(h/2), w), (255, 255, 255), 1)
    out_img = cv2.line(out_img, (0, int(w/2)), (h, int(w/2)), (255, 255, 255), 1)
    out_img = cv2.rectangle(out_img, (0, 0), (h, w), (255, 255, 255), 3)
    out_img = cv2.putText(out_img, '3D pose', (int(w/2)-20, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    if interpreter_3dpose:
        rot_img = np.zeros((w, h, 3), dtype=np.uint8) + 160
        rot_img = cv2.line(rot_img, (int(h/2), 0), (int(h/2), w), (255, 255, 255), 1)
        rot_img = cv2.line(rot_img, (0, int(w/2)), (h, int(w/2)), (255, 255, 255), 1)
        rot_img = cv2.rectangle(rot_img, (0, 0), (h, w), (255, 255, 255), 3)
        rot_img = cv2.putText(rot_img, '3D pose (rotate {} degree)'.format(rot), (int(w/2)-50, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        concat_img = np.hstack((out_img, rot_img))

    for pose in poses:
        if pose.score < min_pose_score: continue
        points = [(-1., -1.)] * 17
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < min_part_score: continue
            # Coord
            kp_y = keypoint.yx[0]
            kp_x = keypoint.yx[1]
            points[KEYPOINTS[label]] = (int(kp_x), int(kp_y))

        # draw 3D pose
        points = [np.array(vec) for vec in points]
        points = to36M(points, BODY_PARTS)
        points = np.reshape(points, [1, -1]).astype('f')
        out_img = pose_util.draw_pose(points[0], out_img)
        if interpreter_3dpose:
            # get rotation pose
            points_norm = normalize_2d(points)
            pose = create_pose(interpreter_3dpose, points_norm)
            rot_img = pose_util.create_projection_pose(pose, np.pi * rot / 180., rot_img)
    if interpreter_3dpose:
        concat_img = np.hstack((out_img, rot_img))
        return concat_img

    return out_img


if __name__ == '__main__':
    main()
