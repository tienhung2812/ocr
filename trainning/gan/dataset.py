from __future__ import print_function, division

from keras.datasets import mnist
(X_train, y_train), (_, _) = mnist.load_data()

print(X_train[0])