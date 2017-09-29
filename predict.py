import os
import json
import logging
import data_helper
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

# def predict_unseen_data():
"""Step 0: load trained model and parameters"""
params = json.loads(open('./parameters.json').read())
checkpoint_dir = "./runs/1506663323/"
if not checkpoint_dir.endswith('/'):
	checkpoint_dir += '/'
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
logging.critical('Loaded the trained model: {}'.format(checkpoint_file))

"""Step 1: load data for prediction"""

# test_file = sys.argv[2]
# test_examples = json.loads(open(test_file).read())

test_variants_file = "Data/stage2_test_variants" # Data source for the training data.
test_text_file = "Data/stage2_test_text" # Data source for the test data.
x_test, y_test, labels = data_helper.load_data_and_labels(test_variants_file,test_text_file)

# x_raw = [example['consumer_complaint_narrative'] for example in test_examples]
# x_test = [data_helper.clean_str(x) for x in x_raw]
logging.info('The number of x_test: {}'.format(len(x_test)))

vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_test)))

"""Step 2: compute the predictions"""
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	sess = tf.Session(config=session_conf)

	with sess.as_default():
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		input_x = graph.get_operation_by_name("input_x").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		predictions = graph.get_operation_by_name("output/predictions").outputs[0]
		probabilities = graph.get_operation_by_name("output/probabilities").outputs[0]

		batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
		all_predictions = []
		# all_probabilities = []
		all_probabilities = pd.DataFrame([])
		for x_test_batch in batches:
			batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
			all_predictions = np.concatenate([all_predictions, batch_predictions])
			batch_probabilities = sess.run(probabilities, {input_x: x_test_batch, dropout_keep_prob: 1.0})

			all_probabilities = all_probabilities.append(pd.DataFrame(batch_probabilities))
		all_probabilities.columns = ['class'+str(c+1) for c in range(9)]
		all_probabilities.insert(0,'ID',[i for i in range(all_probabilities.shape[0])])
		all_probabilities.to_csv('submission.csv', index=False)
if y_test is not None:
	y_test = np.argmax(y_test, axis=1)
	correct_predictions = sum(all_predictions == y_test)
	logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))

# if __name__ == '__main__':
# 	# python3 predict.py ./trained_model_1478649295/ ./data/small_samples.json
# 	predict_unseen_data()