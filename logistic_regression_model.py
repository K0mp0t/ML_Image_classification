import numpy as np
from model import read_dataset_from_h5, save_params_to_file, sigmoid


def initialize_parameters(m):
    W = np.random.randn(m, 1)*0.01
    b = 0

    return W, b


def propagate(X, Y, W, b):
    m = X.shape[1]

    A = sigmoid(np.dot(W.T, X)+b)[0]

    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    dW = (1/m)*np.dot(X, (A-Y).T)
    db = (1/m)*np.sum(A-Y)

    grads = {'dW': dW,
             'db': db}
    cost = np.squeeze(cost)

    return grads, cost



def model(X, Y, learning_rate, num_iterations, print_cost=False):
    W, b = initialize_parameters(X.shape[0])

    for i in range(num_iterations+1):
        grads, cost = propagate(X, Y, W, b)

        W = W - learning_rate*grads['dW']
        b = b - learning_rate*grads['db']

        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: ' % i, cost)

    return W, b


def predict(X, W, b):
    m = X.shape[1]
    Y_predicted = np.zeros((1, m))
    W = W.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(W.T, X)+b)[0]

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_predicted[0, i] = 1
        else:
            Y_predicted[0, i] = 0
    return Y_predicted


if __name__ == '__main__':
    train_x, train_y = read_dataset_from_h5(r'E:/Peter/data_big.h5', 'train')
    test_x, test_y = read_dataset_from_h5(r'E:/Peter/data_big.h5', 'test')
    W, b = model(train_x.T, train_y.T, 0.001, 3000, print_cost=True)
    Y_predicted = predict(train_x.T, W, b)
    print('train accuracy: {} %'.format(100 - np.mean(np.abs(Y_predicted - train_y.T)) * 100))
    Y_predicted = predict(test_x.T, W, b)
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(Y_predicted - test_y.T)) * 100))
