import numpy as np
import random


class NeuralNetwork:

    def __init__(self, layerSizes):
        self.numLayers = len(layerSizes)
        self.layerSizes = len(layerSizes)
        self.weights = [np.random.randn(m, n) for n, m in zip(layerSizes[:-1], layerSizes[1:])]
        self.biases = [np.random.randn(n, 1) for n in layerSizes[1:]]
        self.learningRate = 0.1

    def stepForward(self, matrix, x, b):
        return np.dot(matrix, x) + b

    def stepForwardWithActivation(self, matrix, x, b):
        return activation(np.dot(matrix, x) + b)

    def predict(self, input):
        a = input
        for w, b in zip(self.weights, self.biases):
            a = self.stepForwardWithActivation(w, a, b)
        return a

    def backprop(self, x, y):
        deltaBiases = [np.zeros(bias.shape) for bias in self.biases]
        deltaWeights = [np.zeros(weight.shape) for weight in self.weights]

        a = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = self.stepForward(w, a, b)
            zs.append(z)
            a = activation(z)
            activations.append(a)

        cost = activations[-1] - y
        delta = cost * activationPrimed(zs[-1])

        deltaBiases[-1] = delta
        deltaWeights[-1] = np.dot(delta, activations[-2].transpose())

        for i in range(2, self.numLayers):
            z = zs[-i]
            sp = activationPrimed(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            deltaBiases[-i] = delta
            deltaWeights[-i] = np.dot(delta, activations[-i - 1].transpose())
        return deltaWeights, deltaBiases

    def processBatch(self, batch):
        newBiases = [np.zeros(bias.shape) for bias in self.biases]
        newWeights = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in batch:
            deltaW, deltaB = self.backprop(x, y)
            newWeights = [nw + dw for nw, dw in zip(newWeights, deltaW)]
            newBiases = [nb + db for nb, db in zip(newBiases, deltaB)]

        self.weights = [w - (self.learningRate / len(batch)) * nw for w, nw in zip(self.weights, newWeights)]
        self.biases = [b - (self.learningRate / len(batch)) * nb for b, nb in zip(self.biases, newBiases)]

    def SGD(self, training_data, epochs, mini_batch_size):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.processBatch(mini_batch)

            print("Epoch {0} complete".format(j))


def activation(x):
    return 1.0 / (1.0 + np.exp(-x))


def activationPrimed(x):
    return activation(x) * (1 - activation(x))
