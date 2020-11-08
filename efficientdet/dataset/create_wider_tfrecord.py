#original /content/automl/efficientdet/dataset/create_pascal_tfrecord.py
#modified /content/automl/efficientdet/dataset/create_wider_tfrecord.py

"""Convert WiderFace dataset to TFRecord.

Example usage:
    python create_faces_tfrecord.py  --data_dir=''  --output_path=/tmp/wider
"""
import hashlib
import io
import json
import os

from absl import app
from absl import flags
from absl import logging

import PIL.Image
import tensorflow as tf

from dataset import tfrecord_util

from wider_anno import widerface_annotations


flags.DEFINE_string('data_dir', '', 'Root directory to raw Faces_DD dataset.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', '',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord and json.')
flags.DEFINE_integer('num_shards', 40, 'Number of shards for output file.')
flags.DEFINE_integer('num_images', None, 'Max number of imags to process.')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']


label_map_dict = {
    'background': 0,
    'face': 1,
}

GLOBAL_IMG_ID = 0  # global image id.
GLOBAL_ANN_ID = 0  # global annotation id.


def get_image_id():
  """Convert a string to a integer."""
  # COCO needs int values, here we just use a incremental global_id, but
  # users should customize their own ways to generate filename.
  global GLOBAL_IMG_ID
  GLOBAL_IMG_ID += 1
  return GLOBAL_IMG_ID


def get_ann_id():
  """Return unique annotation id across images."""
  global GLOBAL_ANN_ID
  GLOBAL_ANN_ID += 1
  return GLOBAL_ANN_ID


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ann_json_dict=None):
  """
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: from wider_annotations[SET]
    dataset_directory: Path to root directory holding WIDER dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the  dataset
      directory holding the actual image data.
    ann_json_dict: annotation json dictionary.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  
  img_path = os.path.join(data['path'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(image.width)
  height = int(image.height)
  image_id = get_image_id()
  if ann_json_dict:
    image = {
        'file_name': data['path'],
        'height': height,
        'width': width,
        'id': image_id,
    }
    ann_json_dict['images'].append(image)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  area = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []

  for i,bx in enumerate(data['bbox']):
    difficult = False

    difficult_obj.append(int(difficult))

    xmin.append(float(bx[0]) / width)
    ymin.append(float(bx[1]) / height)
    xmax.append(float(bx[2]) / width)
    ymax.append(float(bx[3]) / height)
    area.append((xmax[-1] - xmin[-1]) * (ymax[-1] - ymin[-1]))
    classes_text.append('face'.encode('utf8'))
    classes.append(label_map_dict['face'])
    truncated.append(False)
    poses.append(str(data['poses'][i]).encode('utf8'))

    if ann_json_dict:
      abs_xmin = int(bx[0])
      abs_ymin = int(bx[1])
      abs_xmax = int(bx[2])
      abs_ymax = int(bx[3])
      abs_width = abs_xmax - abs_xmin
      abs_height = abs_ymax - abs_ymin
      ann = {
          'area': abs_width * abs_height,
          'iscrowd': 0,
          'image_id': image_id,
          'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
          'category_id': label_map_dict['face'],
          'id': get_ann_id(),
          'ignore': 0,
          'segmentation': [],
      }
      ann_json_dict['annotations'].append(ann)

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height':
                  tfrecord_util.int64_feature(height),
              'image/width':
                  tfrecord_util.int64_feature(width),
              'image/filename':
                  tfrecord_util.bytes_feature(str(data['path']).encode('utf8')),
              'image/source_id':
                  tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
              'image/key/sha256':
                  tfrecord_util.bytes_feature(key.encode('utf8')),
              'image/encoded':
                  tfrecord_util.bytes_feature(encoded_jpg),
              'image/format':
                  tfrecord_util.bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin':
                  tfrecord_util.float_list_feature(xmin),
              'image/object/bbox/xmax':
                  tfrecord_util.float_list_feature(xmax),
              'image/object/bbox/ymin':
                  tfrecord_util.float_list_feature(ymin),
              'image/object/bbox/ymax':
                  tfrecord_util.float_list_feature(ymax),
              'image/object/area':
                  tfrecord_util.float_list_feature(area),
              'image/object/class/text':
                  tfrecord_util.bytes_list_feature(classes_text),
              'image/object/class/label':
                  tfrecord_util.int64_list_feature(classes),
              'image/object/difficult':
                  tfrecord_util.int64_list_feature(difficult_obj),
              'image/object/truncated':
                  tfrecord_util.int64_list_feature(truncated),
              'image/object/view':
                  tfrecord_util.bytes_list_feature(poses),
          }))
  return example


def main(_):
  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.format(SETS))
  if not FLAGS.output_path:
    raise ValueError('output_path cannot be empty.')

  data_dir = FLAGS.data_dir


  output_dir = os.path.dirname(FLAGS.output_path)
  if not tf.io.gfile.exists(output_dir):
    tf.io.gfile.makedirs(output_dir)
  logging.info('Writing to output directory: %s', output_dir)

  for SET, Faces in widerface_annotations().items():
    print(SET, len(Faces))


    writers = [
        tf.io.TFRecordWriter(FLAGS.output_path + '_' + SET + '-%05d-of-%05d.tfrecord' %
                                    (i, FLAGS.num_shards))
        for i in range(FLAGS.num_shards)
    ]


    ann_json_dict = {
        'images': [],
        'type': 'instances',
        'annotations': [],
        'categories': []
    }


    for class_name, class_id in label_map_dict.items():
      cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
      ann_json_dict['categories'].append(cls)

    logging.info('Reading from WIDER ' + SET + ' dataset.')
    for idx, example in enumerate(Faces):
      if FLAGS.num_images and idx >= FLAGS.num_images:
        break
      if idx % 1000 == 0:
        logging.info('On image %d of %d', idx, len(Faces))

      tf_example = dict_to_tf_example(
          Faces[idx],
          FLAGS.data_dir,
          label_map_dict,
          ann_json_dict=ann_json_dict)
      writers[idx % FLAGS.num_shards].write(tf_example.SerializeToString())

    for writer in writers:
      writer.close()

    json_file_path = os.path.join(
        os.path.dirname(FLAGS.output_path),
        'json_' + SET +'_' + os.path.basename(FLAGS.output_path) + '.json')
    with tf.io.gfile.GFile(json_file_path, 'w') as f:
      json.dump(ann_json_dict, f)


if __name__ == '__main__':
  app.run(main)
