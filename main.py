from nn import NeuralNetwork
import numpy as np
from keras import datasets
from keras.utils import to_categorical

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255

x_train = np.array(x_train)
x_train = [np.reshape(x, (784, 1)) for x in x_train]
y_train = np.array(to_categorical(y_train))
y_train = [np.reshape(y, (10, 1)) for y in y_train]

nn = NeuralNetwork([784, 32, 32, 10])
nn.SGD(np.array(list((zip(x_train, y_train)))), 10, 32)

ctr = 0
for i in range(0, 100):
    res = nn.predict(x_train[i])
    print("guessed: {}, correct: {}".format(np.argmax(res), np.argmax(y_train[i])))
    if np.argmax(res) == np.argmax(y_train[i]):
        ctr += 1
print("{}/{}".format(ctr, 100))

res = nn.predict(x_train[0])
