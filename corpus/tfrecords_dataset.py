import os
import sys
import pdb
import json
import pickle
import numpy as np
import tensorflow as tf
import google.protobuf.json_format
# import util_tfrecords from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import src.util_tfrecord as util_tfrecord

import torch 

def prepare_wave(wave):
    wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
    normalized_waves = []
    for i in range(3):
        normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
        normalized_waves.append(normalized_wave)
    wave = tf.stack(normalized_waves, axis=0)
    wave = tf.cast(wave, tf.float32)
    return wave


def read_labeled_tfrecord(example):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_wave(example["wave"]), tf.reshape(tf.cast(example["target"], tf.float32), [1]), example["wave_id"]


def read_unlabeled_tfrecord(example, return_image_id):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_wave(example["wave"]), example["wave_id"] if return_image_id else 0


def get_dataset(files, batch_size=16, buffer_size_shuffle=100, repeat=False, cache=False, shuffle=False, labeled=True, return_image_ids=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(buffer_size_shuffle)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids), num_parallel_calls=AUTO)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return tfds.as_numpy(ds)


class TFRecordDataLoader:
    def __init__(self, files, batch_size=32, cache=False, train=True, repeat=False, shuffle=False, labeled=True, return_image_ids=True):
        self.ds = get_dataset(
            files, 
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,
            shuffle=shuffle,
            labeled=labeled,
            return_image_ids=return_image_ids)
        
        if train:
            self.num_examples = count_data_items(files)
        else:
            self.num_examples = count_data_items_test(files)

        self.batch_size = batch_size
        self.labeled = labeled
        self.return_image_ids = return_image_ids
        self._iterator = None
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        n_batches = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1