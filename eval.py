#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("test_variants_file", "Data/test_variants", "Data source for the training data.")
tf.flags.DEFINE_string("test_text_file", "Data/test_text", "Data source for the test data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 30,  "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1506408531/checkpoints", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data. 

x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.test_variants_file, FLAGS.test_text_file)

try:
    y_test = np.argmax(y_test, axis=1)
except:
    pass

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
ts1 = vocab_processor.transform(x_raw)
ts1 = list(ts1)
x_test = np.array(ts1)

index = np.array(range(x_test.shape[0]))

# Evaluation
# ==================================================
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        print ("\nSession restored...\n")

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        print ("Retrived the graph...\n")
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        print ("Retrived the dropout parameter...\n")

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        print ("Retrived prediction tensor...\n")

        probabilities = graph.get_operation_by_name("output/probabilities").outputs[0]
        print ("Retrived probabilities tensor...\n")


        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
        print ("Retrived the batch parameter...\n")

        # Collect the predictions here
        all_predictions = []
        all_probabilities = []

        for x_test_batch in batches:
            batch_predictions = sess.run(probabilities, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_probabilities = np.concatenate([all_probabilities, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
#predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
b = np.array2string(all_predictions, precision=2)
        
predictions_human_readable = np.column_stack((list(index), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    column = ['ID']
    columns = ['class'+str(c+1) for c in range(9)]
    columns=column + columns
    csv.writer(f).writerow(columns)
    csv.writer(f).writerows(predictions_human_readable)