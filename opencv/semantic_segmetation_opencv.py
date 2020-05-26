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
r"""An example using `BasicEngine` to perform semantic segmentation.

The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:

python3 examples/semantic_segmentation.py \
--model models/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
--input models/bird.bmp \
--keep_aspect_ratio \
--output ${HOME}/segmentation_result.jpg
"""

import argparse
import platform
import subprocess
from edgetpu.segmentation.engine import SegmentationEngine
from edgetpu.utils import dataset_utils, image_processing
from PIL import Image
from PIL import ImageDraw
import numpy as np
import cv2
import os


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  indices = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((indices >> channel) & 1) << shift
    indices >>= 3

  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      help='Path of the segmentation model.',
      required=True)
  parser.add_argument(
      '--input', help='File path of the input image.', required=True)
  parser.add_argument('--output', help='File path of the output image.')
  parser.add_argument(
      '--keep_aspect_ratio',
      dest='keep_aspect_ratio',
      action='store_true',
      help=(
          'keep the image aspect ratio when down-sampling the image by adding '
          'black pixel padding (zeros) on bottom or right. '
          'By default the image is resized and reshaped without cropping. This '
          'option should be the same as what is applied on input images during '
          'model training. Otherwise the accuracy may be affected and the '
          'bounding box of detection result may be stretched.'))
  parser.set_defaults(keep_aspect_ratio=False)
  args = parser.parse_args()

  # Initialize engine.
  engine = SegmentationEngine(args.model)
  _, height, width, _ = engine.get_input_tensor_shape()

  # Read frame from camera (or video)
  cap = cv2.VideoCapture(args.input)

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2_im = frame

    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2_im_rgb)
    # Open image.
    if args.keep_aspect_ratio:
        resized_img, ratio = image_processing.resampling_with_original_ratio(
            img, (width, height), Image.NEAREST)
    else:
        resized_img = img.resize((width, height))
        ratio = (1., 1.)

    input_tensor = np.asarray(resized_img).flatten()
    _, raw_result = engine.run_inference(input_tensor)
    result = np.reshape(raw_result, (height, width))
    new_width, new_height = int(width * ratio[0]), int(height * ratio[1])

    # If keep_aspect_ratio, we need to remove the padding area.
    result = result[:new_height, :new_width]
    vis_result = label_to_color_image(result.astype(int)).astype(np.uint8)
    vis_result = Image.fromarray(vis_result)

    vis_img = resized_img.crop((0, 0, new_width, new_height))
    # Concat resized input image and processed segmentation results.
    concated_image = Image.new('RGB', (new_width*2, new_height))
    concated_image.paste(vis_img, (0, 0))
    concated_image.paste(vis_result, (width, 0))

    #cv2_im = append_objs_to_img(cv2_im, vis_result)
    concated_image = np.array(concated_image)
    concated_image = concated_image[:, :, ::-1].copy()
    cv2.imshow('frame', concated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, objs):
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
  main()
