from __future__ import print_function

import os


import functools
import pandas as pd
from sklearn.model_selection import train_test_split


import tensorflow.contrib as tfcontrib

import tensorflow as tf

class Data_Preprocess(object):

    def __init__(self, **kwargs):
        """
        Arguments:
        - dataset_path
        - train_path
            -imgage_path
            -mask_path
        - test_path
            -image_path 
        - csv_path
        """
        self.data_path =  kwargs.pop('dataset_path', os.path.join( os.getcwd() ,   "DATA/dataset"))
        self.train_path = kwargs.pop('train_path', os.path.join( self.data_path , "training"))
        self.test_path =  kwargs.pop('test_path', os.path.join( self.data_path ,   "testing"))
        self.csv_path =   kwargs.pop('csv_path', os.path.join( self.data_path ,     "train.txt"))
        self.train_im_dir =  os.path.join(self.train_path, "images")
        self.train_mask_dir = os.path.join(self.train_path, "labels")

        # Optional params
        self.img_shape = kwargs.pop('image_shape', (64, 64, 3))  #(256, 256, 3)
        self.batch_size = kwargs.pop('batch_size', 1)  # 3
        self.threads = kwargs.pop('threads', 5)
        
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)


    def get_train_val_split_paths(self):
             #print(self.csv_path)
             ids_train = pd.read_csv(self.csv_path, sep='\n', header=None)[0]
             x_train_filenames = []
             y_train_filenames = []
             for img_id in ids_train:
                x_train_filenames.append(os.path.join(self.train_im_dir, "{}".format(img_id)))
                y_train_filenames.append(os.path.join(self.train_mask_dir, "{}".format(img_id)))
             x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = train_test_split(x_train_filenames, y_train_filenames, test_size=0.2, random_state=42)

             num_train_examples = len(x_train_filenames)
             num_val_examples = len(x_val_filenames)

             print("Number of training examples: {}".format(num_train_examples))
             print("Number of validation examples: {}".format(num_val_examples))
             return (x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames)



    def _process_pathnames(self,fname, label_path):
        # We map this function onto each pathname pair
        img_str = tf.read_file(fname)
        img = tf.image.decode_png(img_str, channels=3)

        label_img_str = tf.read_file(label_path)
        # These are gif images so they return as (num_frames, h, w, c)
        label_img = tf.image.decode_png(label_img_str, channels=3)  # [0]
        # The label image should only have values of 1 or 0, indicating pixel wise
        # object (car) or not (background). We take the first channel only.

        label_img = label_img[:, :, 1]
        label_img = tf.expand_dims(label_img, axis=-1)
        return img, label_img


    def shift_img(self,output_img, label_img, width_shift_range, height_shift_range):
        """This fn will perform the horizontal or vertical shift"""
        if width_shift_range or height_shift_range:
            if width_shift_range:
                width_shift_range = tf.random_uniform([],
                                                      -width_shift_range * self.img_shape[1],
                                                      width_shift_range * self.img_shape[1])
            if height_shift_range:
                height_shift_range = tf.random_uniform([],
                                                       -height_shift_range * self.img_shape[0],
                                                       height_shift_range * self.img_shape[0])
            # Translate both
            output_img = tfcontrib.image.translate(output_img,
                                                   [width_shift_range, height_shift_range])
            label_img = tfcontrib.image.translate(label_img,
                                                  [width_shift_range, height_shift_range])
        return output_img, label_img

    def flip_img(self, horizontal_flip, tr_img, label_img):
        if horizontal_flip:
            flip_prob = tf.random_uniform([], 0.0, 1.0)
            tr_img, label_img = tf.cond(tf.less(flip_prob, 0.5),
                                        lambda: (tf.image.flip_left_right(tr_img), tf.image.flip_left_right(label_img)),
                                        lambda: (tr_img, label_img))
        return tr_img, label_img

    def _augment(self, img,
                 label_img,
                 resize=None,  # Resize the image to some size e.g. [256, 256]
                 scale=1,  # Scale image e.g. 1 / 255.
                 hue_delta=0,  # Adjust the hue of an RGB image by random factor
                 horizontal_flip=False,  # Random left right flip,
                 width_shift_range=0,  # Randomly translate the image horizontally
                 height_shift_range=0):  # Randomly translate the image vertically
        if resize is not None:
            # Resize both images
            label_img = tf.image.resize_images(label_img, resize)
            img = tf.image.resize_images(img, resize)

        if hue_delta:
            img = tf.image.random_hue(img, hue_delta)

        img, label_img = self.flip_img(horizontal_flip, img, label_img)
        img, label_img = self.shift_img(img, label_img, width_shift_range, height_shift_range)
        label_img = tf.to_float(label_img) * scale
        img = tf.to_float(img) * scale
        return img, label_img

    def get_baseline_dataset(self, filenames,
                             labels,
                             preproc_fn=functools.partial(_augment),
                             shuffle=True):
        num_x = len(filenames)
        # Create a dataset from the filenames and labels
        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # Map our preprocessing function to every element in our dataset, taking
        # advantage of multithreading
        dataset = dataset.map(self._process_pathnames, num_parallel_calls=self.threads)
        if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
            assert self.batch_size == 1, "Batching images must be of the same size"

        dataset = dataset.map(preproc_fn, num_parallel_calls=self.threads)

        if shuffle:
            dataset = dataset.shuffle(num_x)

        # It's necessary to repeat our data for all epochs
        dataset = dataset.repeat().batch(self.batch_size)
        return dataset