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
import csv
import glob
import numpy as np
from PIL import Image
from pose_engine import PoseEngine

engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
pose_score_thresh = 0.4
# csv writer
f = open('pose_output.csv', 'w')
with f:
    fnames = ['timestamp', 'idx', 'label', 'width', 'height', 'x', 'y', 'score']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

    # read frames
    for image_path in sorted(glob.glob('/home/mendel/dataset/Store/frames/Camera01/*.jpg')):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        #print(image_name)
        pil_image = Image.open(image_path)
        pil_image.resize((641, 481), Image.NEAREST)
        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
        #print('Inference time: %.fms' % inference_time)

        # save pose
        idx = -1
        for pose in poses:
            if pose.score < 0.4: continue
            #print('\nPose Score: ', pose.score)
            idx += 1
            for label, keypoint in pose.keypoints.items():
                writer.writerow({'timestamp' : image_name, 'idx': idx, 'label': label, 'width': '641', 'height': '481', 'x': keypoint.yx[1], 'y': keypoint.yx[0], 'score': keypoint.score})
