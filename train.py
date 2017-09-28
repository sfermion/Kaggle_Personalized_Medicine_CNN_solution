#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: amin
references:
    https://github.com/jiegzhan/multi-class-text-classification-cnn/blob/master/
    https://github.com/dennybritz/cnn-text-classification-tf/blob/master/
"""
import datetime
import data_helper
import json
import logging
import tensorflow as tf
import time
import numpy as np
import os

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from text_cnn import TextCNN
#from sklearn.preprocessing import normalize

logging.getLogger().setLevel(logging.INFO)

# Parameters
# ==================================================

# Data loading params
training_variants_file = "Data/training_variants" # Data source for the training data.
training_text_file = "Data/training_text" # Data source for the test data.

parameter_file = "./parameters.json"
params = json.loads(open(parameter_file).read())

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_raw, y_raw, labels = data_helper.load_data_and_labels(training_variants_file,training_text_file)


# Build vocabulary
max_document_length = max([len(set(x.split(" "))) for x in x_raw])
logging.info('The maximum length of all sentences: {}'.format(max_document_length))
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
ts = vocab_processor.fit_transform(x_raw)
ts = list(ts)
x = np.array(ts)
y = np.array(y_raw)

# index = np.array(range(len(y)))
# x = np.column_stack((index,x))

x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# x_test_idx = x_test[:,0]
# x_test = x_test[:,1:]

# x_idx = x_[:,0]
# x_ = x_[:,1:]
# Randomly shuffle data

shuffle_indices = np.random.permutation(np.arange(len(y_)))
x_shuffled = x_[shuffle_indices]
y_shuffled = y_[shuffle_indices]

x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

# x_train_idx = x_train[:,0]
# x_train = x_train[:,1:]
# x_dev_idx = x_dev[:,0]
# x_dev = x_dev[:,1:]


with open('./labels.json', 'w') as outfile:
    json.dump(labels, outfile, indent=4)

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))



# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]

# # Split train/test set

# dev_sample_index = -1 * int(eval_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


# Training
# ==================================================

# build a graph and cnn object
    
logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

graph = tf.Graph()
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement= params['allow_soft_placement'],
      log_device_placement= params['log_device_placement'])
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=params['embedding_dim'],
            filter_sizes=list(map(int, params['filter_sizes'].split(","))),
            num_filters=params['num_filters'],
            l2_reg_lambda=params['l2_reg_lambda'])

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(5e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)


        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep= params['num_checkpoints'])

        # One training step: train the model with one batch
        def train_step(x_batch, y_batch):

            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: params['dropout_keep_prob']
            }
            _, step, summaries, loss, accuracy = sess.run([train_op, 
                                                        global_step, 
                                                        train_summary_op, 
                                                        cnn.loss, 
                                                        cnn.accuracy],
                                                        feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        # One evaluation step: evaluate the model with one batch
        def dev_step(x_batch, y_batch):
            feed_dict = { cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
            step, loss, accuracy, num_correct = sess.run([global_step, 
                                                        cnn.loss, 
                                                        cnn.accuracy, 
                                                        cnn.num_correct],
                                                        feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # if writer:
            #     writer.add_summary(summaries, step)

            return num_correct

        # Save the word_to_id map since predict.py needs it
        vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
        sess.run(tf.global_variables_initializer())

        # Generate batches
        batches = data_helper.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
        best_accuracy, best_at_step = 0, 0

        """ Train the cnn model with x_train and y_train (batch by batch)"""
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            """Evaluate the model with x_dev and y_dev (batch by batch)"""
            if current_step % params['evaluate_every'] == 0:
                dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)),params['batch_size'], 1)
                total_dev_correct = 0
                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                    total_dev_correct += num_dev_correct


                dev_accuracy = float(total_dev_correct) / len(y_dev)
                logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                """Save the model if it is the best based on accuracy of the dev set"""
                if dev_accuracy >= best_accuracy:
                    best_accuracy, best_at_step = dev_accuracy, current_step
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                    logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

                # print("\nEvaluation:")
                # dev_step(x_dev, y_dev, writer=dev_summary_writer)
                # print("")
            # if current_step % checkpoint_every == 0:
                # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                # print("Saved model checkpoint to {}\n".format(path))

        """ Predict x_test (batch by batch)"""
        test_batches = data_helper.batch_iter(list(zip(x_test, y_test)),params['batch_size'], 1)
        total_test_correct = 0
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            num_test_correct = dev_step(x_test_batch, y_test_batch)
            total_test_correct += num_test_correct

        test_accuracy = float(total_test_correct) / len(y_test)
        logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
        logging.critical('The training is complete')
