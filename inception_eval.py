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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
#############################1
from inception import inception_PCA


#'/home/frank/last_ubuntu_download/models-update-models-1.0/inception/home/frank/last_ubuntu_download/models-update-models-1.0/inception/inception/place_train',
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation' or 'train'.""")


def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, endpoints):#V,S,mean_x):############################2
  """Runs Eval once.
  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Successfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      count_top_5 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      #########################6
      V_ = []
      S_ = []
      count = 0
      E_team = []
      V_last = []
      X_list = []
      nda = 10

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        top_1, top_5 = sess.run([top_1_op, top_5_op])
        count_top_1 += np.sum(top_1)
        count_top_5 += np.sum(top_5)

        ###############################################7
        # collect input of endpoints
        if not count % nda:
          E_team = np.reshape(endpoints['mixed_8x8x2048b'].eval(), (32,131072))
        else:
          E_team = np.append(E_team, np.reshape(endpoints['mixed_8x8x2048b'].eval(), (32,131072)))
        print("E_team.shape:", E_team.shape)
        count += 1
        if not count % nda:
          print("E_team.shape:", tf.shape(E_team))
          E_team_team = np.reshape(E_team, (32*nda,131072))
          # compute PCA and print it
          #sampless = np.random.normal(size = (8,16))
          #print(sampless)
          #VV, reco_X, e = inception_PCA._pca(sampless, 8)

          # compute autocorrelation
          MAX_coeff_k, mean_coeff, coeff_auto, X = inception_PCA._find_max_autocorrelation(E_team_team, 32*nda, 100,250 )
          print(MAX_coeff_k)
          print(mean_coeff)
          print(coeff_auto)
          X_list = [X_list,E_team]
          E_team = []


        step += 1
        if step % 20 == 0:

          duration = time.time() - start_time
          sec_per_batch = duration / 20.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      ###############################################8
      #print(_compute_cross_correlation(X_list, 20, 200, 2000))

      # Compute precision @ 1.
      precision_at_1 = count_top_1 / total_sample_count
      recall_at_5 = count_top_5 / total_sample_count
      print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
            (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels = image_processing.distorted_inputs(dataset)##4

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ , endpoints = inception.inference(images, num_classes)

    # print end_points['mixed_8x8x2048b']###########################################3
    #sss = {}
    #print(sss = tf.add(endpoints['mixed_8x8x2048b'][0],endpoints['mixed_8x8x2048b'][1] ))
    #array_end_points = endpoints['mixed_8x8x2048b'].eval()
    #V, S, mean_x = inception_PCA._pca(endpoints['mixed_8x8x2048b'])

    # Calculate predictions.
    top_1_op = tf.nn.in_top_k(logits, labels, 1)
    top_5_op = tf.nn.in_top_k(logits, labels, 5)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                            graph_def=graph_def)

    while True:
      _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, endpoints)#V,S,mean_x)#5
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
