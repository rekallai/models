# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
A library to train Inception using multiple GPU's with synchronous updates, by
calculating the gradients of the network in subbatches making it feasible to use large batch sizes
on GPUs with small memory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim

from inception import tfconnect as tfc

import sys

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_integer('num_sub_batches_per_batch', 1,
                            """Number of sub-batches per batch.""")

tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

REUSE_VARIABLES = None


def get_new_image_and_label_batch_splits(
        dataset,
        batch_size,
        num_gpus=FLAGS.num_gpus,
        num_preprocess_threads_per_gpu=FLAGS.num_preprocess_threads):
    """
    
    Returns new image label batch split in sub-batches per GPU
    
    :param dataset: 
    :param batch_size: 
    :param num_gpus: 
    :param num_preprocess_threads_per_gpu: 
    :return: 
    """

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert batch_size % num_gpus == 0, (
        '(Sub) Batch size must be divisible by number of GPUs')

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = num_preprocess_threads_per_gpu * num_gpus
    images, labels = image_processing.distorted_inputs(
        dataset,
        batch_size,
        num_preprocess_threads=num_preprocess_threads)

    # Split the batch of images and labels for towers.
    images_splits = tf.split(axis=0, num_or_size_splits=num_gpus, value=images)
    labels_splits = tf.split(axis=0, num_or_size_splits=num_gpus, value=labels)

    return images_splits, labels_splits


def calc_gradients(opt, images_splits, labels_splits, num_classes):
    # ugh, I know
    global REUSE_VARIABLES

    # Calculate the gradients for each model tower.
    tower_grads = []
    batchnorm_updates = []
    loss = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
                # Force all Variables to reside on the CPU.
                with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                    # Calculate the loss for one tower of the ImageNet model. This
                    # function constructs the entire ImageNet model but shares the
                    # variables across all towers.
                    batch_loss = _tower_loss(images_splits[i], labels_splits[i], num_classes,
                                       scope, REUSE_VARIABLES)

                REUSE_VARIABLES = True

                batchnorm_updates.append(tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION, scope))

                loss.append(batch_loss)

                # Calculate the gradients for the batch of data on this ImageNet
                # tower.
                grads = opt.compute_gradients(batch_loss)

                # Keep track of the gradients across all towers.
                tower_grads.append(grads)

    return tower_grads, batchnorm_updates, loss


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
    """Calculate the total loss on a single tower running the ImageNet model.
  
    We perform 'batch splitting'. This means that we cut up a batch across
    multiple GPU's. For instance, if the batch size = 32 and num_gpus = 2,
    then each tower will operate on an batch of 16 images.
  
    Args:
      images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                         FLAGS.image_size, 3].
      labels: 1-D integer Tensor of [batch_size].
      num_classes: number of classes
      scope: unique prefix string identifying the ImageNet tower, e.g.
        'tower_0'.
  
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # When fine-tuning a model, we do not restore the logits but instead we
    # randomly initialize the logits. The number of classes in the output of the
    # logit is the number of classes in specified Dataset.
    restore_logits = not FLAGS.fine_tune

    # Build inference Graph.
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits = inception.inference(images, num_classes, for_training=True,
                                     restore_logits=restore_logits,
                                     scope=scope)

    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    split_batch_size = images.get_shape().as_list()[0]
    inception.loss(logits, labels, batch_size=split_batch_size)

    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

    # Calculate the total loss for the current tower.
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on TensorBoard.
        loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(loss_name + ' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))

    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
  
    Note that this function provides a synchronization point across all towers.
  
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def train(dataset):
    """Train on dataset for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                                 FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        FLAGS.learning_rate_decay_factor,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)

        sub_batch_size = int(FLAGS.batch_size / FLAGS.num_sub_batches_per_batch)

        images_splits, labels_splits = get_new_image_and_label_batch_splits(
            dataset,
            sub_batch_size,
            num_gpus=FLAGS.num_gpus,
            num_preprocess_threads_per_gpu=FLAGS.num_preprocess_threads)

        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = dataset.num_classes() + 1

        get_tower_grads, get_batchnorm_updates, calc_loss = calc_gradients(
            opt,
            images_splits,
            labels_splits,
            num_classes)

        # Create the placeholder structure to connect all the subbatch gradient data to the averaging step
        tower_grads_list_placeholder = []
        counter = -1
        is_variable = dict()
        for sub_batch_idx in range(FLAGS.num_sub_batches_per_batch):
            tower_grads_placeholder, counter, is_variable = tfc.create_placeholder_for(
                get_tower_grads,
                'tower_grads',
                counter,
                is_variable)
            tower_grads_list_placeholder += tower_grads_placeholder

        calc_averaged_gradient = average_gradients(tower_grads_list_placeholder)

        # Apply the gradients to adjust the shared variables.
        apply_gradient = opt.apply_gradients(calc_averaged_gradient, global_step=global_step)

        # Group all updates to into a single train op.
        # batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
        
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        tf.train.start_queue_runners(sess=sess)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            subbatch_gradients = []
            for sub_batch_idx in range(FLAGS.num_sub_batches_per_batch):
                tower_grads, batchnorm_updates = sess.run((get_tower_grads, get_batchnorm_updates))
                subbatch_gradients += tower_grads
                sys.stdout.write('*')
            print()

            av_gradient_feed, _ = tfc.create_feed_based_on(subbatch_gradients, 'tower_grads', is_variable)

            sess.run(apply_gradient, av_gradient_feed)
            loss_values = sess.run(calc_loss)
            av_loss = sum(loss_values) / float(len(loss_values))

            duration = time.time() - start_time

            assert not np.isnan(av_loss), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, av_loss,
                                    examples_per_sec, duration))

            # Save the model checkpoint periodically.
            if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

