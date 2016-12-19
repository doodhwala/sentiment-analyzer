import numpy as np
import csv
from gensim.models import Word2Vec

model = Word2Vec.load('model')

def relu(z):
	return z * (z > 0)

def relu_prime(z):
	return (z > 0)

class RNN:
	def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, direction="right"):
		self.direction = direction
		self.hidden_size = hidden_size 
		self.learning_rate = learning_rate

		self.Wxh = np.random.randn(hidden_size, input_size) * 0.01 
		self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
		self.Why = np.random.randn(output_size, hidden_size) * 0.01 
		self.bh = np.zeros((hidden_size, 1)) 
		self.by = np.zeros((output_size, 1)) # output bias - computed but not used

		self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad

	def forward(self, inputs, hprev):
		if(self.direction == 'left'):
			inputs = inputs[::-1]
	
		xs, hs, ys, ps = {}, {}, {}, {}
		hs[-1] = np.copy(hprev)

		for t in range(len(inputs)):
			xs[t] = inputs[t].reshape(-1, 1)
			hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) 
			ys[t] = np.tanh(np.dot(self.Why, hs[t]) + self.by)
			ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
		return xs, hs, ys, ps

	def backprop(self, xs, hs, ys, ps, targets, dy):
		if(self.direction == 'left'):
			xs = {len(xs)-1-k : xs[k] for k in xs}

		dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
		dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
		dhnext = np.zeros_like(hs[-1])
		for t in reversed(range(len(xs))):
			tmp = dy[t] * (1 - ys[t] ** 2)
			#tmp = dy[t] * relu_prime(ys[t])
			dWhy += np.dot(tmp, hs[t].T)
			dby += tmp
			dh = np.dot(self.Why.T, dy[t]) + dhnext
			dhraw = dh * (1 - hs[t] ** 2)
			#dhraw = dh * relu_prime(hs[t])
			dbh += dhraw
			dWxh += np.dot(dhraw, xs[t].T)
			dWhh += np.dot(dhraw, hs[t-1].T)
			dhnext = np.dot(self.Whh.T, dhraw)

		for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
			np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

		return dWxh, dWhh, dWhy, dbh, dby, hs[len(xs)-1]

	def update_params(self, dWxh, dWhh, dWhy, dbh, dby):
		# perform parameter update with Adagrad
		for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
		                            [dWxh, dWhh, dWhy, dbh, dby],
		                            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
			mem += dparam * dparam
			param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

class BiDirectionalRNN:
	def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
		self.hidden_size = hidden_size
		self.learning_rate = learning_rate

		self.right = RNN(input_size, hidden_size, output_size, learning_rate, direction="right")
		self.left = RNN(input_size, hidden_size, output_size, learning_rate, direction="left")

		self.by = np.zeros((output_size, 1))
		self.mby = np.zeros_like(self.by)

	def forward(self, x):
		seq_length = len(x)
		if seq_length == 0:
			return 2
			# make it decent
		y_pred = []
		dby = np.zeros_like(self.by)
		xsl, hsl, ysl, psl = self.left.forward(x, np.zeros((self.hidden_size, 1)))
		xsr, hsr, ysr, psr = self.right.forward(x, np.zeros((self.hidden_size, 1)))

		for ind in range(seq_length):
			this_y = np.dot(self.right.Why, hsr[ind]) + np.dot(self.left.Why, hsl[ind]) + self.by
			y_pred.append(this_y)
		return np.argmax(y_pred[-1])

	def train(self, inputs, targets, epochs=5):
		for e in range(epochs):
			print('Epoch #{}'.format(e + 1))

			for x, y in zip(inputs, targets):
				hprevr = np.zeros((self.hidden_size, 1))
				hprevl = np.zeros((self.hidden_size, 1))
					
				
				seq_length = len(x)

				xsl, hsl, ysl, psl = self.left.forward(x, hprevl)
				xsr, hsr, ysr, psr = self.right.forward(x, hprevr)

				y_pred = []
				dy = []
				dby = np.zeros_like(self.by)
				for ind in range(seq_length):
					this_y = np.dot(self.right.Why, hsr[ind]) + np.dot(self.left.Why, hsl[ind]) + self.by
					y_pred.append(this_y)

				for ind in range(seq_length):
					this_dy = np.exp(y_pred[ind]) / np.sum(np.exp(y_pred[ind]))
					t = np.argmax(y)
					#t = y
					this_dy[t] -= 1
					dy.append(this_dy)
					dby += this_dy

				y_pred = np.array(y_pred)
				dy = np.array(dy)

				self.mby += dby * dby
				self.by += -self.learning_rate * dby / np.sqrt(self.mby + 1e-8) # adagrad update

				dWxhr, dWhhr, dWhyr, dbhr, dbyr, hprevr = self.right.backprop(xsr, hsr, ysr, psr, y, dy)
				dWxhl, dWhhl, dWhyl, dbhl, dbyl, hprevl = self.left.backprop(xsl, hsl, ysl, psl, y, dy)

				self.right.update_params(dWxhr, dWhhr, dWhyr, dbhr, dbyr)
				self.left.update_params(dWxhl, dWhhl, dWhyl, dbhl, dbyl)

		print("Training done.")

	def predict(self, inputs, targets):
		correct = 0
		predictions = {x:0 for x in range(3)}
		outputs = {x:0 for x in range(3)}
		for x, y in zip(inputs, targets):
			# ip = [np.argmax(i) for i in x]
			op = self.forward(x)
			tr = np.argmax(y)
			# I have changed this
			# print op, tr
			predictions[op] += 1
			outputs[tr] += 1
			correct += 1 if op == tr else 0
		print 'Outputs: ', outputs
		print 'Predictions:', predictions
		return (correct + 0.0) / len(inputs)

def three(x):
	if(x < 2):
		return 0
	elif(x > 2):
		return 2
	else:
		return 1

def load_data(filename):
	i = 0
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		inputs = []
		outputs = []
		for row in reader:
			inputs.append(row[0])
			outputs.append(int(row[1]))
			i += 1
			if i == 2500:
				break
		return inputs, outputs

def w2v(sentence):
	words = []
	for word in sentence:
		try:
			words.append(model[word])
		except Exception as e:
			pass
	# print sentence
	# print words
	return np.array(words)

# OUTPUT_SIZE
def ohv(x):
	v = np.zeros(3)
	v[three(x)] = 1
	return v

if __name__ == "__main__":
	INPUT_SIZE = 32
	HIDDEN_SIZE = 16
	OUTPUT_SIZE = 3
	
	training_inputs, training_targets =  load_data('train.csv')
	testing_inputs, testing_targets =  load_data('test.csv')
	for i in range(len(training_inputs)):
		training_inputs[i] = w2v(training_inputs[i])
	for i in range(len(training_targets)):
		training_targets[i] = ohv(training_targets[i])
	for i in range(len(testing_inputs)):
		testing_inputs[i] = w2v(testing_inputs[i])
	for i in range(len(testing_targets)):
		testing_targets[i] = ohv(testing_targets[i])


	EPOCHS = 10
	LEARNING_RATE = 0.33

	BRNN = BiDirectionalRNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, learning_rate=LEARNING_RATE)

	BRNN.train(training_inputs, training_targets, epochs=EPOCHS)

	accuracy = BRNN.predict(testing_inputs, testing_targets)

	print("Accuracy: {}%".format(accuracy * 100))

'''

three_way_ip = np.array([three(x) for x in ip])
three_way_op = np.array([three(x) for x in op])
just need to change output size

Create two BiDiRNNs
'''