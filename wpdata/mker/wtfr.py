from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, random, os, shutil, sys
from math import floor, ceil
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

from wputils.utils.utils import pe, pt
from wputils.utils.io import rnii


def spread_lists(lists, num_shards=8):
    nested_list = []
    instance_per_shards = ceil(len(lists) / num_shards)
    for i in range(num_shards):
        dir_list = lists[i * instance_per_shards: (i + 1) * instance_per_shards]
        nested_list_tmp = [dir_list, i]
        nested_list.append(nested_list_tmp)

    return nested_list


def tfrecords_writer(nested_list, total_shards, tfrecords_base_name):
    filename = os.path.join(tfrecords_base_name, '%.3d_of_%.3d' % (nested_list[1], total_shards) + '.tfrecords')
    writer = tf.io.TFRecordWriter(filename)

    for file_dir in nested_list[0]:
        try:
            img_arr, stk = rnii(os.path.join(file_dir, 'img.nii.gz'))
            seg_arr, _ = rnii(os.path.join(file_dir, 'seg.nii.gz'))

            img_arr = img_arr.astype(np.int16)
            seg_arr = seg_arr.astype(np.uint8)

            assert tuple(img_arr.shape) == (12, 480, 480) and tuple(seg_arr.shape) == (12, 480, 480)

            # assert tuple(img_arr.shape) == (48, 480, 480) and tuple(seg_arr.shape) == (48, 480, 480)

            dim = list(img_arr.shape)
            dim = np.array(dim).astype(np.uint8)

            origin = np.array(stk.GetOrigin()).astype(np.float32)
            direction = np.array(stk.GetDirection()).astype(np.float32)
            spacing = np.array(stk.GetSpacing()).astype(np.float32)

            example = tf.train.Example(features=tf.train.Features(feature={
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_arr.tobytes()])),
                'seg': tf.train.Feature(bytes_list=tf.train.BytesList(value=[seg_arr.tobytes()])),
                'dim': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dim.tobytes()])),
                'origin': tf.train.Feature(bytes_list=tf.train.BytesList(value=[origin.tobytes()])),
                'direction': tf.train.Feature(bytes_list=tf.train.BytesList(value=[direction.tobytes()])),
                'spacing': tf.train.Feature(bytes_list=tf.train.BytesList(value=[spacing.tobytes()])),
            }))
            writer.write(example.SerializeToString())
        except Exception as e:
            print(f'file processed failed on: {file_dir}')
            print('-' * 60)
            print(pe(e))
            print('-' * 60)
            continue
    writer.close()
    print('working done on file: {}'.format(filename))
    sys.stdout.flush()


def npy2tf_writer(folder_dir_list, tfrecords_base_name, num_shards=8, is_mtt=False, cpu_num=None):
    nested_lists = spread_lists(folder_dir_list, num_shards=num_shards)

    if is_mtt:
        cpu_num = num_shards if cpu_num is None else cpu_num
        p = Pool(cpu_num)
        p.starmap(tfrecords_writer, [(nested_list, num_shards, tfrecords_base_name) for nested_list in nested_lists])
        p.close()
    else:
        for nested_list in nested_lists:
            tfrecords_writer(nested_list, num_shards, tfrecords_base_name)

    return None
