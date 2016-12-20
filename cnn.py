import numpy as np
import scipy
import scikit-learn
import csv
import sys

from gensim.models import Word2Vec

model = Word2Vec.load('model64')

def clip(sentence):
		return np.array(sentence[:10])

""" ------------------------------------------------------------------------------- """

class ConvolutionalNeuralNet:
	def __init__(self, filter_shape):
		self.W = np.random.randn(*filter_shape)
		pass

	def forward(self, x):
		print x.shape
		exit()
		x = scipy.signal.convolve2d(x, self.W) )

	def train(self, training_data, validation_data, epochs=5):):
		for x, y in zip(*training_data):
			x = clip(x)

	def predict(self, testing_data, test=False):
		correct = 0
		predictions = {x : 0 for x in range(3)}
		outputs = {x : 0 for x in range(3)}
		l = 0
		for x, y in zip(*testing_data):
			if len(x) == 0:
				continue

			op = self.forward(x)
			tr = np.argmax(y)
			predictions[op] += 1
			outputs[tr] += 1
			correct += 1 if op == tr else 0
			l += 1

		if test:
			print 'Outputs:\t', outputs
			print 'Predictions:\t', predictions

		return (correct + 0.0) / l

""" ------------------------------------------------------------------------------- """

def load_data(filename, count):
	i = 0
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		inputs = []
		outputs = []
		for row in reader:
			inputs.append(row[0])
			outputs.append(int(row[1]))
			i += 1
			if i == count:
				break
		return inputs, outputs

def w2v(sentence):
	words = []
	for word in sentence.split():
		try:
			words.append(model[word])
		except Exception:
			pass

	return np.array(words)

def one_hot(x):
	v = np.zeros(5)
	v[x] = 1
	return v

if __name__ == "__main__":
	DATA_SIZE = 20000

	INPUT_SIZE = 64
	HIDDEN_SIZE = 16
	OUTPUT_SIZE = 5
	
	train_size = DATA_SIZE * 0.8
	val_size = DATA_SIZE * 0.1
	test_size = DATA_SIZE * 0.1
	
	t_i, t_t =  load_data('train.csv', train_size)
	v_i, v_t = load_data('dev.csv', val_size)
	ts_i, ts_t =  load_data('test.csv', test_size)

	training_inputs = []
	training_targets = []
	for i in range(len(t_i)):
		v = w2v(t_i[i])
		if len(v) == 0:
			continue

		training_inputs.append(v)
		training_targets.append(one_hot(t_t[i]))

	validation_inputs = []
	validation_targets = []
	for i in range(len(v_i)):
		v = w2v(v_i[i])
		if len(v) == 0:
			continue

		validation_inputs.append(v)
		validation_targets.append(one_hot(v_t[i]))

	testing_inputs = []
	testing_targets = []
	for i in range(len(ts_i)):
		v = w2v(ts_i[i])
		if len(v) == 0:
			continue

		testing_inputs.append(v)
		testing_targets.append(one_hot(ts_t[i]))

	EPOCHS = 1
	LEARNING_RATE = 0.20

	CNN = ConvolutionalNeuralNet()

	CNN.train(training_data=(training_inputs, training_targets), validation_data=(validation_inputs, validation_targets), epochs=EPOCHS)

	accuracy = CNN.predict((testing_inputs, testing_targets), True)

	print("Accuracy: {:.2f}%".format(accuracy * 100))








