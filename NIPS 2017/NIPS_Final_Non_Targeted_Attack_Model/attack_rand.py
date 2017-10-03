"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import Attack, FastGradientMethod
import numpy as np
from PIL import Image
from cleverhans.model import Model, CallableModelWrapper
from cleverhans.utils_tf import *
import tensorflow as tf
from tensorflow.contrib.slim.nets import inception
import inception_resnet_v2
import random

random.seed(3228)

class MadryEtAl(Attack):

    """
    The Projected Gradient Descent Attack (Madry et al. 2016).
    Paper link: https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a MadryEtAl instance.
        """
        super(MadryEtAl, self).__init__(model, back, sess)
        self.feedable_kwargs = {'eps': np.float32,
                                                        'eps_iter': np.float32,
                                                        'y': np.float32,
                                                        'y_target': np.float32,
                                                        'clip_min': np.float32,
                                                        'clip_max': np.float32}
        self.structural_kwargs = ['ord', 'nb_iter']

        if not isinstance(self.model, Model):
                self.model = CallableModelWrapper(self.model, 'probs')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                                compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                                         y_target=None if y is also set. Labels should be
                                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                                Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x)

        return adv_x

    def parse_params(self, eps=0.3, eps_iter=0.01, nb_iter=40, y=None,
                                     ord=np.inf, clip_min=None, clip_max=None,
                                     y_target=None, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.
        Attack-specific parameters:
        :param eps: (required float) maximum distortion of adversarial example
                                compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                                         y_target=None if y is also set. Labels should be
                                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                                Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max

        if self.y is not None and self.y_target is not None:
                raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
                raise ValueError("Norm order must be either np.inf, 1, or 2.")
        if self.back == 'th':
                error_string = ("ProjectedGradientDescentMethod is"
                                                " not implemented in Theano")
                raise NotImplementedError(error_string)

        return True

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        

        adv_x = x + eta
        preds = self.model.get_probs(adv_x)
        loss = model_loss(y, preds)
        if self.targeted:
                loss = -loss
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return x, eta

    def attack(self, x, **kwargs):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.
        :param x: A tensor with the input image.
        """

        eta = tf.random_uniform(tf.shape(x), -self.eps, self.eps)
        eta = clip_eta(eta, self.ord, self.eps)

        if self.y is not None:
                y = self.y
        else:
                preds = self.model.get_probs(x)
                preds_max = tf.reduce_max(preds, 1, keep_dims=True)
                y = tf.to_float(tf.equal(preds, preds_max))
                y = y / tf.reduce_sum(y, 1, keep_dims=True)
        y = tf.stop_gradient(y)

        for i in range(self.nb_iter):
                x, eta = self.attack_single_step(x, eta, y)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
                adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x


slim = tf.contrib.slim


tf.flags.DEFINE_string(
        'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
        'resnet_checkpoint_path', 'inception_resnet_v2_2016_08_30.ckpt', 'Path to checkpoint for inception network.')
        
tf.flags.DEFINE_string(
        'inception_checkpoint_path', 'inception_v3.ckpt', 'Path to checkpoint for inception network.')
        
tf.flags.DEFINE_string(
        'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
        'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
        'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
        'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
        'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
        'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

files_inception = []
#files_resnet = []
files_madry_inception = []
files_madry_resnet = []

def random_split_image(input_dir):
    global files_inception, files_madry_inception, files_madry_resnet
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        rd = random.randint(1,3000)
        if(rd <= 1000):
            files_inception.append(filepath)
        elif(rd <= 2000):
            files_madry_resnet.append(filepath)
        else:
            files_madry_inception.append(filepath)
            
def load_images(filepathes, batch_shape):
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
    for filepath in filepathes:
        with tf.gfile.Open(filepath,'rb') as f:
            image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
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


def save_images(images, filenames, output_dir):
    """Saves images to the output directory.

    Args:
        images: array with minibatch of images
        filenames: list of filenames without path
            If number of file names in this list less than number of images in
            the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
    """
    for i, filename in enumerate(filenames):
        # Images for inception classifier are normalized to be in [-1, 1] interval,
        # so rescale them back to [0, 1].
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')

class InceptionModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                    x_input, num_classes=self.num_classes, is_training=False,
                    reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs
            
class InceptionResnetModel(object):
    """Model class for CleverHans library."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            _, end_points = inception_resnet_v2.inception_resnet_v2(
                    x_input, num_classes=self.num_classes, is_training=False,
                    reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        # Strip off the extra reshape op at the output
        probs = output.op.inputs[0]
        return probs


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)
    
    global files_inception, files_madry_inception, files_madry_resnet
    random_split_image(FLAGS.input_dir)
    
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        resnet_model = InceptionResnetModel(num_classes)
        madry_resnet = MadryEtAl(resnet_model)
                
        
        x_adv = madry_resnet.generate(x_input, eps=eps, eps_iter=eps/8, nb_iter=12, clip_min=-1., clip_max=1.)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_filename_with_path=FLAGS.resnet_checkpoint_path,
                master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(files_madry_resnet, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
    
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        
        inception_model = InceptionModel(num_classes)
        madry_inception = MadryEtAl(inception_model)
        
        
        x_adv = madry_inception.generate(x_input, eps=eps, eps_iter=eps/8, nb_iter=12, clip_min=-1., clip_max=1.)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
                scaffold=tf.train.Scaffold(saver=saver),
                checkpoint_filename_with_path=FLAGS.inception_checkpoint_path,
                master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(files_madry_inception, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        inception_model = InceptionModel(num_classes)

        fgsm = FastGradientMethod(inception_model)
        x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.inception_checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(files_inception, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
                
    '''            
    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        resnet_model = InceptionResnetModel(num_classes)

        fgsm = FastGradientMethod(resnet_model)
        x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

        # Run computation
        saver = tf.train.Saver(slim.get_model_variables())
        session_creator = tf.train.ChiefSessionCreator(
            scaffold=tf.train.Scaffold(saver=saver),
            checkpoint_filename_with_path=FLAGS.resnet_checkpoint_path,
            master=FLAGS.master)

        with tf.train.MonitoredSession(session_creator=session_creator) as sess:
            for filenames, images in load_images(files_resnet, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)
    '''

if __name__ == '__main__':
    tf.app.run()
