"""Implementation of sample defense.

This defense loads inception resnet v2 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread

import tensorflow as tf

import inception_resnet_v2
from tensorflow.contrib.slim.nets import inception
import operator





tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
        'resnet_checkpoint_path', 'inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')
        
tf.flags.DEFINE_string(
        'adv_inception_checkpoint_path', 'adv_inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
        'vgg_checkpoint_path', 'vgg_16.ckpt', 'Path to checkpoint for inception network.')
        
tf.flags.DEFINE_string(
        'inception_checkpoint_path', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
        'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
        'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
        'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
        filenames: list file names without path of each image
            Lenght of this list could be less than batch_size, in this case only
            first few images of the result are elements of the minibatch.
        images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        with tf.gfile.Open(filepath, 'rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

    
def main(_):
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    file_names = []
    resnet_preds = {}
    adv_inception_preds = {}
    inception_preds = {}
    vgg_preds = {}
    
    tf.logging.set_verbosity(tf.logging.INFO)
    g1 = tf.Graph()
    g2 = tf.Graph()
    g3 = tf.Graph()
    g4 = tf.Graph()
    
    with g1.as_default():
        # Prepare graph
        slim = tf.contrib.slim
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(
                    x_input, num_classes=num_classes, is_training=False)

            # Run computation
            saver = tf.train.Saver(slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                    scaffold=tf.train.Scaffold(saver=saver),
                    checkpoint_filename_with_path=FLAGS.resnet_checkpoint_path,
                    master=FLAGS.master)

            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={x_input: images})
                        
                    for filename, predict_value, logit_value in zip(filenames, predict_values, logit_values):
                        resnet_preds[filename] = predict_value
                        file_names.append(filename)
            print("resnet pred complete")
    tf.reset_default_graph()    
    with g2.as_default():
        # Prepare graph
        slim = tf.contrib.slim
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                    x_input, num_classes=num_classes, is_training=False)
                    
            # Run computation
            saver = tf.train.Saver(slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                    scaffold=tf.train.Scaffold(saver=saver),
                    checkpoint_filename_with_path=FLAGS.adv_inception_checkpoint_path,
                    master=FLAGS.master)

            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={x_input: images})
                        
                    for filename, predict_value, logit_value in zip(filenames, predict_values, logit_values):
                        adv_inception_preds[filename] = predict_value
            
            print("adv_inception pred complete")
    tf.reset_default_graph()
    with g3.as_default():
        # Prepare graph
        slim = tf.contrib.slim
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                    x_input, num_classes=num_classes, is_training=False)
                    
            # Run computation
            saver = tf.train.Saver(slim.get_model_variables())
            session_creator = tf.train.ChiefSessionCreator(
                    scaffold=tf.train.Scaffold(saver=saver),
                    checkpoint_filename_with_path=FLAGS.inception_checkpoint_path,
                    master=FLAGS.master)

            with tf.train.MonitoredSession(session_creator=session_creator) as sess:
                for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={x_input: images})
                        
                    for filename, predict_value, logit_value in zip(filenames, predict_values, logit_values):
                        inception_preds[filename] = predict_value
            
            print("inception pred complete")            
        
        with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
            for filename in file_names:
                ensemble_pred = resnet_preds[filename] * 0.34 + \
                                adv_inception_preds[filename] * 0.33 + inception_preds[filename] * 0.33
                out_file.write('{0},{1}\n'.format(filename, np.argmax(ensemble_pred)))
            
        
if __name__ == '__main__':
    tf.app.run()
