import sys
sys.path.append('../')

import numpy as np
import gradient_descent as gd

# Python implementation
def model(x, w, b):
    return w @ x + b

def cost(x_train, y_train, w, b):
    # Mean-squared error
    return 1/(2*x_train.shape[0]) * np.sum(
        np.square([
            model(x, w, b) - y_train[i]
            for i, x in enumerate(x_train)
        ])
    )

def gradients(x_train, y_train, w, b):
    return gd.gradients(model, x_train, y_train, w, b)

def fit(x_train, y_train, w, b, learning_rate, num_iter, epsilon = None, log_interval = 100):
    return gd.fit(cost, gradients, x_train, y_train, w, b, learning_rate, num_iter, epsilon, log_interval)

