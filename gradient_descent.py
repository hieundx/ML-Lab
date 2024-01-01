import numpy as np

def gradients(model_fn, x_train, y_train, w, b):
    m = x_train.shape[0]
    dj_dw = 1/m * np.sum(
        [
            (model_fn(x, w, b) - y_train[i]) * x 
            for i, x in enumerate(x_train)
        ],
        0 # set dimension to 0. numpy will sum all matrixes and return a matrix instead of scalar
    )

    dj_db = 1/m * np.sum(
        [
            model_fn(w, x, b) - y_train[i]
            for i, x in enumerate(x_train)
        ]
    )

    return dj_dw, dj_db

def fit(cost_fn, gradients_fn, x_train, y_train, w, b, learning_rate, num_iter, epsilon = None, log_interval = 100):
    costs = []
    for i in range(num_iter):
        dj_dw, dj_db = gradients_fn(x_train, y_train, w, b)

        w -= learning_rate * dj_dw
        b -= learning_rate * dj_db

        cost = cost_fn(x_train, y_train, w, b)
        
        costs.append(cost)

        if epsilon != None and cost < epsilon:
            print (f'Converged after {i} iteration')
            break
        
        if i % log_interval == 0:
            print(f'[{i}] {cost}')

    return costs, w, b

