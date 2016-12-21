import numpy as np
import scipy
import sklearn
import dill
import pickle
import csv
import sys
import copy

from gensim.models import Word2Vec

model = Word2Vec.load('model64')

def clip(v):
		x = v[:10]
		if len(x) % 2 == 0:
			b = a = (10 - len(x)) / 2
		else:
			b = (10 - len(x)) / 2
			a = b + 1

		return np.lib.pad(np.array(x), ((b, a), (0, 0) ), 'constant')

def save_model(brnn):
	with open('cnn_model_%s.pkl' % TYPE, 'wb') as f:
		dill.dump(brnn, f)

def load_model():
	with open('cnn_model_%s.pkl' % TYPE, 'rb') as f:
		brnn = dill.load(f)
	return brnn

""" ------------------------------------------------------------------------------- """

class ConvolutionalNeuralNet:
	def __init__(self, vec_size, input_dim, filter_dim, num_filters, stride, output_size, learning_rate=0.01):



		# input_size = 10
		# filter_size = 2
		# stride = 1

		# output_size = (input_size - filter_size + 1) / stride

		# x = np.random.randn(input_size * 64)
		# y = np.random.randn(filter_size * 64)

		# o = []

		# 
		# i, j = 0, 0
		# while j < output_size:
		# 	o.append(np.convolve(x[i : i + 128], y, 'valid')) 
		# 	i += stride * 64
		# 	j += 1

		# print len(np.array(o))

		self.num_filters = num_filters
		self.stride = stride
		self.vec_size = vec_size
		self.input_dim = input_dim	
		self.filter_dim = filter_dim
		self.result_size = (input_dim - filter_dim) / stride + 1
		self.learning_rate = learning_rate
		self.f = np.tanh
		self.f_prime = lambda x: 1 - (x ** 2)

		self.Wxh = np.random.randn(self.num_filters, self.filter_dim * vec_size) * np.sqrt(2.0 / (self.filter_dim * vec_size + 1))
		self.Why = np.random.randn(output_size, num_filters ) * np.sqrt(2.0 / (output_size + num_filters))
		self.bh = np.zeros((self.num_filters, self.result_size))
		self.by = np.zeros((output_size, 1))

		self.mWxh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)
		self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad	

	def forward(self, x):
		h = []
		mask = []
		for f, b in zip(self.Wxh, self.bh):
			c = []
			indices = []
			i, j = 0, 0
			while j < self.result_size:
				c.append(np.dot(x[i : i + self.filter_dim].reshape(-1, ), f) + b[j])
				indices.append((i, i + self.filter_dim))
				j += 1
				i += self.stride

			h.append(np.max(c))
			mask.append([np.argmax(c)] + list(indices[np.argmax(c)]) )

		y = self.f(np.dot(self.Why, np.array(h).reshape(-1, 1)) + self.by)
		p = np.exp(y) / np.sum(np.exp(y))

		return np.array(h), np.array(mask), y, p


	def backprop(self, x, h, mask, y, dy):
		dWxh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

		tmp = dy * self.f_prime(y)
		dWhy = np.dot(tmp, h.reshape(-1, 1).T)
		dby = tmp
		dh = np.dot(self.Why.T, dy)
		dhraw = dh * self.f_prime(h.reshape(-1, 1))

		for i, d in enumerate(dhraw):
			dWxh[i] = np.dot( x[mask[i][1] : mask[i][2]].reshape(-1, 1), d.reshape(-1, 1)).reshape(-1)
			dbh[i][mask[i][0]] = d * self.bh[i][mask[i][0]]

		for dparam in [dWxh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
		
		return dWxh, dWhy, dbh, dby
		
	def train(self, training_data, validation_data, epochs=5):
		for e in range(epochs):
			print('Epoch {}'.format(e + 1))

			for x, target_y in zip(*training_data):
				h, mask, y, p = self.forward(x)
				t = np.argmax(target_y)
				dy = copy.copy(p)
				dy[t] -= 1
				dWxh, dWhy, dbh, dby = self.backprop(x, h, mask, y, dy)
				self.update_params(dWxh, dWhy, dbh, dby)

			print("(val acc: {:.2f}%)".format(self.predict(validation_data) * 100))

		print("\nTraining done.")

	def update_params(self, dWxh, dWhy, dbh, dby):
		# perform parameter update with Adagrad
		for param, dparam, mem in zip([self.Wxh, self.Why, self.bh, self.by],
		                            [dWxh, dWhy, dbh, dby],
		                            [self.mWxh, self.mWhy, self.mbh, self.mby]):
			mem += dparam * dparam
			param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

	def predict(self, testing_data, test=False):
		if testing_data[1] == None:
			predictions = []
			for x in testing_data[0]:
				op = self.forward(x)
				predictions.append(np.argmax(y))

			return predictions

		else:
			correct = 0
			predictions = {x : 0 for x in range(TYPE)}
			outputs = {x : 0 for x in range(TYPE)}

			l = 0
			for x, y in zip(*testing_data):
				op = np.argmax(self.forward(x)[-1])
				tr = np.argmax(y)
				predictions[op] += 1
				outputs[tr] += 1
				correct = correct + 1 if op == tr else correct + 0
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
	def three(x):
		if x < 2:
			return 0
		
		elif x > 2:
			return 2
		
		else:
			return 1

	v = np.zeros(TYPE)

	if TYPE == 3:
		v[three(x)] = 1
	else:
		v[x] = 1
	
	return v

if __name__ == "__main__":
	DATA_SIZE = 5000
	TYPE = 3

	INPUT_SIZE = 64
	FILTER_DIM = 2
	NUM_FILTERS = 16
	STRIDE = 1
	POOL_DIM = 2
	OUTPUT_SIZE = TYPE

	
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

		training_inputs.append(clip(v))
		training_targets.append(one_hot(t_t[i]))

	validation_inputs = []
	validation_targets = []
	for i in range(len(v_i)):
		v = w2v(v_i[i])
		if len(v) == 0:
			continue

		validation_inputs.append(clip(v))
		validation_targets.append(one_hot(v_t[i]))

	testing_inputs = []
	testing_targets = []
	for i in range(len(ts_i)):
		v = w2v(ts_i[i])
		if len(v) == 0:
			continue

		testing_inputs.append(clip(v))
		testing_targets.append(one_hot(ts_t[i]))

	EPOCHS = 3
	LEARNING_RATE = 0.01

	TRAIN = True

	CNN = None
	if TRAIN:
		CNN = ConvolutionalNeuralNet(INPUT_SIZE, 10, FILTER_DIM, NUM_FILTERS, STRIDE, OUTPUT_SIZE, LEARNING_RATE)
		CNN.train(training_data=(training_inputs, training_targets), validation_data=(validation_inputs, validation_targets), epochs=EPOCHS)
		save_model(CNN)
	else:
		CNN = load_model()
	

	accuracy = CNN.predict((testing_inputs, testing_targets), True)

	print("Accuracy: {:.2f}%".format(accuracy * 100))







