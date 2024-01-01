import numpy as np

# Python implementation
def f(x, w, b):
    return w @ x + b

def j(x_train, y_train, w, b):
    # Mean-squared error
    return 1/(2*x_train.shape[0]) * np.sum(
        np.square([
            f(x, w, b) - y_train[i]
            for i, x in enumerate(x_train)
        ])
    )

def gradients(x_train, y_train, w, b):
    m = x_train.shape[0]
    dj_dw = 1/m * np.sum(
        [
            (f(x, w, b) - y_train[i]) * x 
            for i, x in enumerate(x_train)
        ],
        0 # set dimension to 0. numpy will sum all matrixes and return a matrix instead of scalar
    )

    dj_db = 1/m * np.sum(
        [
            f(w, x, b) - y_train[i]
            for i, x in enumerate(x_train)
        ]
    )

    return dj_dw, dj_db

def fit(x_train, y_train, w, b, learning_rate, num_iter, epsilon = None, log_interval = 100):
    costs = []
    for i in range(num_iter):
        dj_dw, dj_db = gradients(x_train, y_train, w, b)

        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        cost = j(x_train, y_train, w, b)
        
        costs.append(cost)

        if epsilon != None and cost < epsilon:
            print (f'Converged after {i} iteration')
            break
        
        if i % log_interval == 0:
            print(f'[{i}] {cost}')

    return costs, w, b

