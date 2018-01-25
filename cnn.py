import numpy as np
import scipy
import dill
import csv

from gensim.models import Word2Vec

model = Word2Vec.load('model64')


def relu(z):
    return z * (z > 0)


def relu_prime(z):
    return (z > 0)


def clip(v):
    x = v[:10]
    if len(x) % 2 == 0:
        b = a = (10 - len(x)) / 2
    else:
        b = (10 - len(x)) / 2
        a = b + 1

    return np.lib.pad(np.array(x), ((b, a), (0, 0)), 'constant')


def save_model(brnn):
    with open('cnn_models/cnn_model_%s.pkl' % TYPE, 'wb') as f:
        dill.dump(brnn, f)


def load_model():
    with open('cnn_models/cnn_model_%s.pkl' % TYPE, 'rb') as f:
        brnn = dill.load(f)
    return brnn


""" ------------------------------------------------------------------------------- """


class ConvolutionalNeuralNet:
    def __init__(self, filter_dim, num_filters, output_size, learning_rate=0.01):
        N, h, w = (num_filters, (10 - filter_dim) + 1, (64 - filter_dim) + 1)  # result of convolution
        self.result_shape = (N, h / 2, w / 2)  # result of pooling
        self.filter_shape = (num_filters, filter_dim, filter_dim)
        self.input_shape = (10, 64)

        self.learning_rate = learning_rate
        self.f = np.tanh
        self.f_prime = lambda x: 1 - (x ** 2)
        self.Wxh = np.random.randn(*self.filter_shape) * np.sqrt(2.0 / (sum(self.filter_shape)))
        self.Why = np.random.randn(output_size, np.prod(self.result_shape)) * np.sqrt(
            2.0 / (np.prod(self.filter_shape) + output_size))
        self.bh = np.zeros((num_filters, h, w))
        self.by = np.zeros((output_size, 1))

    def forward(self, x):
        h = self.f(np.array(
            [scipy.signal.convolve2d(x, self.Wxh[i], mode='valid') + self.bh[i] for i in range(len(self.Wxh))]))

        h_prev = np.copy(h)
        num_filters, height, width = h.shape
        h = np.amax(h.reshape(num_filters, height / 2, 2, width / 2, 2).swapaxes(2, 3).reshape(num_filters, height / 2,
                                                                                               width / 2, 4), axis=3)

        mask = np.equal(h_prev, h.repeat(2, axis=1).repeat(2, axis=2)).astype(int)

        y = self.f(np.dot(self.Why, h.reshape(-1, 1)) + self.by)
        p = np.exp(y) / np.sum(np.exp(y))

        return h, mask, y, p

    def backprop(self, x, h, mask, y, dy):
        dWxh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)

        tmp = dy * self.f_prime(y)
        dWhy = np.dot(tmp, h.reshape(-1, 1).T)
        dby = tmp
        dh = np.dot(self.Why.T, dy)
        dhraw = dh * self.f_prime(h.reshape(-1, 1))
        dhraw = dhraw.reshape(self.result_shape).repeat(2, axis=1).repeat(2, axis=2)
        dhraw = np.multiply(dhraw, mask)
        dWxh = np.array([np.rot90(scipy.signal.convolve(x, np.rot90(w, 2), 'valid'), 2) for w in dhraw])
        dbh = dhraw

        # dWxh = np.zeros_like(self.Wxh)
        # dWhy = np.zeros_like(self.Why)
        # dbh = np.zeros_like(self.bh)
        # dby = np.zeros_like(self.by)

        for dparam in [dWxh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        return dWxh, dWhy, dbh, dby

    def train(self, training_data, validation_data, epochs=5):
        for e in range(epochs):
            print('Epoch {}'.format(e + 1))

            for x, target_y in zip(*training_data):
                h, mask, y, p = self.forward(x)
                t = np.argmax(target_y)
                dy = p
                dy[t] -= 1
                dWxh, dWhy, dbh, dby = self.backprop(x, h, mask, y, dy)
                self.update_params(dWxh, dWhy, dbh, dby)

            print("(val acc: {:.2f}%)".format(self.predict(validation_data) * 100))

        print("\nTraining done.")

    def update_params(self, dWxh, dWhy, dbh, dby):
        # perform parameter update with Adagrad
        for param, dparam in zip([self.Wxh, self.Why, self.bh, self.by],
                                 [dWxh, dWhy, dbh, dby]):
            param -= self.learning_rate * dparam

    def predict(self, testing_data, test=False):
        correct = 0
        predictions = {x: 0 for x in range(TYPE)}
        outputs = {x: 0 for x in range(TYPE)}

        pred_pos = {x: 0 for x in range(TYPE)}
        pred_neg = {x: 0 for x in range(TYPE)}

        l = 0
        for x, y in zip(*testing_data):
            op = np.argmax(self.forward(x)[-1])
            tr = np.argmax(y)
            predictions[op] += 1
            outputs[tr] += 1
            correct = correct + 1 if op == tr else correct + 0
            l += 1

            if (op == tr):
                pred_pos[op] += 1
            else:
                pred_neg[op] += 1

        if test:
            print 'Outputs:\t', outputs
            print 'Predictions:\t', predictions
            precision = {}
            recall = {}
            for i in range(TYPE):
                precision[i] = 1 if predictions[i] == 0 else (pred_pos[i] + 0.0) / predictions[i]
                print 'Precision', i, ':', precision[i]
            for i in range(TYPE):
                recall[i] = 1 if outputs[i] == 0 else (pred_pos[i] + 0.0) / (outputs[i])
                print 'Recall', i, ':', recall[i]
            for i in range(TYPE):
                print 'F1 Score', i, ':', (2 * precision[i] * recall[i]) / (precision[i] + recall[i])

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
    DATA_SIZE = 10000
    TYPE = 5

    FILTER_DIM = 3
    NUM_FILTERS = 10
    POOL_DIM = 2
    OUTPUT_SIZE = TYPE

    train_size = DATA_SIZE * 0.8
    val_size = DATA_SIZE * 0.1
    test_size = DATA_SIZE * 0.1

    t_i, t_t = load_data('train.csv', train_size)
    v_i, v_t = load_data('dev.csv', val_size)
    ts_i, ts_t = load_data('test.csv', test_size)

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

    EPOCHS = 5
    LEARNING_RATE = 0.033

    TRAIN = True
    RETRAIN = True

    CNN = None
    if TRAIN:
        if (RETRAIN):
            CNN = load_model()
        else:
            CNN = ConvolutionalNeuralNet(FILTER_DIM, NUM_FILTERS, TYPE, LEARNING_RATE)

        CNN.train(training_data=(training_inputs, training_targets),
                  validation_data=(validation_inputs, validation_targets), epochs=EPOCHS)
        save_model(CNN)
    else:
        CNN = load_model()

    accuracy = CNN.predict((testing_inputs, testing_targets), True)

    print("Accuracy: {:.2f}%".format(accuracy * 100))
