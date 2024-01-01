import sys
sys.path.append('../')

import numpy as np
import gradient_descent as gd

def model(x, w, b):
    sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))

    return sigmoid(w @ x + b)

def loss(x, y, w, b):
    res =  -y * np.log(model(x, w, b)) - (1 - y) * np.log(1 - model(x, w, b))
    return res

def cost(x_train, y_train, w, b):
    return np.mean([
        loss(x_train[i], y_train[i], w, b) for i, _ in enumerate(x_train)
    ])

def gradients(x_train, y_train, w, b):
    return gd.gradients(model, x_train, y_train, w, b)

def fit(x_train, y_train, w, b, learning_rate, num_iter, epsilon = None, log_interval = 100):
    return gd.fit(cost, gradients, x_train, y_train, w, b, learning_rate, num_iter, epsilon, log_interval)

