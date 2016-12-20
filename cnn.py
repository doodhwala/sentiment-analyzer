import numpy
import scipy
import scikit-learn

class ConvolutionalNeuralNet:
	def __init__(self, filter_dim, num_filters):
		self.W = np.random.randn(filter_dim, filter_dim, num_filters)
		pass

	def forward(self, x):
		x = np.array([scipy.signal.convolve(x, w) for w in self.W] )

	def backpropagate(self, output, target):
		pass

	def train(self, inputs, targets):
		for x, y in zip(inputs, targets):
			pass

	def predict(self, inputs, targets):
		pass

if __name__ == "__main__":
	CNN = ConvolutionalNeuralNet()