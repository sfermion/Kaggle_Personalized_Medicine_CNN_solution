#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: amin
references:
    https://github.com/jiegzhan/multi-class-text-classification-cnn/blob/master/
    https://github.com/dennybritz/cnn-text-classification-tf/blob/master/
"""
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

from sklearn.preprocessing import normalize

# Parameters
# ==================================================

# Data loading params
eval_sample_percentage = .3 # Percentage of the training data to use for validation
training_variants_file = "Data/training_variants" # Data source for the training data.
training_text_file = "Data/training_text" # Data source for the test data.


# Model Hyperparameters
embedding_dim = 200 # Dimensionality of character embedding (default: 128)
filter_sizes = "2,3,4" # Comma-separated filter sizes (default: '3,4,5')
num_filters = 256 # Number of filter('positive_data_file's per filter size (default: 128)
dropout_keep_prob = 0.5 # Dropout keep probability (default: 0.5)
l2_reg_lambda = 0.0 # L2 regularization lambda (default: 0.0)


# Training parameters
batch_size = 20 # Batch Size (default: 64)
num_epochs = 100 # Number of training epochs (default: 200)
evaluate_every = 20 # Evaluate model on dev set after this many steps (default: 100)
checkpoint_every = 50 # Save model after this many steps (default: 100)
num_checkpoints = 5 # Number of checkpoints to store (default: 5)


# Misc Parameters
allow_soft_placement = True # Allow device soft device placement
log_device_placement =  False # Log placement of ops on devices


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(training_variants_file,training_text_file)


# Build vocabulary
max_document_length = max([len(set(x.split(" "))) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
ts = vocab_processor.fit_transform(x_text)
ts = list(ts)
x = np.array(ts)

index = np.array(range(len(y)))
x = np.column_stack((index,x))
# x = normalize(x)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set

dev_sample_index = -1 * int(eval_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

index = x_train[:,0]
x_train = x_train[:,1:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

# build a graph and cnn object

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=embedding_dim,
            filter_sizes=list(map(int, filter_sizes.split(","))),
            num_filters=num_filters,
            l2_reg_lambda=l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(5e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep= num_checkpoints)

        vocab_processor.save(os.path.join(out_dir, "vocab"))

        sess.run(tf.global_variables_initializer())

        # One training step: train the model with one batch
        def train_step(x_batch, y_batch):

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        # One evaluation step: evaluate the model with one batch
        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = { cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
            step, loss, accuracy, num_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct],feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return num_correct

        # Save the word_to_id map since predict.py needs it
        vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
        sess.run(tf.global_variables_initializer())

        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
        best_accuracy, best_at_step = 0, 0

        """ Train the cnn model with x_train and y_train (batch by batch)"""
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % evaluate_every == 0:
                dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
                total_dev_correct = 0
                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                    total_dev_correct += num_dev_correct

                dev_accuracy = float(total_dev_correct) / len(y_dev)
                print('Accuracy on dev set: {}'.format(dev_accuracy))



                """ Save the model if it is the best based on accuracy of the dev set"""
                if dev_accuracy >= best_accuracy:
                    best_accuracy, best_at_step = dev_accuracy, current_step
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model {} at step {}'.format(path, best_at_step))
                    print('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

        """ Predict x_test (batch by batch)"""
        test_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), batch_size, 1)
        total_test_correct = 0
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            num_test_correct = dev_step(x_test_batch, y_test_batch)
            total_test_correct += num_test_correct

        test_accuracy = float(total_test_correct) / len(y_dev)
        print('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
        print('The training is complete')
