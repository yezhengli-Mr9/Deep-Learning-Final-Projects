import os
import numpy as np
import tensorflow as tf

def read_labeled_image_list(img_list_path, img_dir, start_index):

  f = open(img_list_path, 'r')
  img_paths = []
  for line in f:
    img_name, index = line[:-1].split(' ')
    img_paths.append(img_dir + img_name)
  f.close()
  return img_paths

def read_images_from_disk(input_queue):
  img_path = tf.read_file(input_queue[0])  
  img = tf.image.decode_png(img_path, channels=1)
  return img


def get_loader(root, batch_size,  start_index, split=None, shuffle=True):

  img_list_path =  root + '/devkit/' + split + '.txt'
  img_dir = root + '/imgs/'

  print('------img dir', img_dir)
  print('------img_list_path', img_list_path)
  img_paths_np = read_labeled_image_list(img_list_path, img_dir, start_index)
  print('img_paths_np: ', img_paths_np)

  with tf.device('/cpu:0'):
    img_paths = tf.convert_to_tensor(img_paths_np, dtype=tf.string)

    input_queue = tf.train.slice_input_producer([img_paths],
                  shuffle=shuffle, capacity=10*batch_size)

    img = read_images_from_disk(input_queue)
    print('finish read_images_from_disk')
    print(img.shape)

    img.set_shape([64, 64, 1])
    img = tf.cast(img, tf.float32)

    img_batch  = tf.train.batch([img], num_threads=1,
                           batch_size=batch_size, capacity=10*batch_size)
    print('finish batch')
    print(img_batch.shape)

  return img_batch, [] #empty label list
