import h5py
import numpy as np
from params_saver import save_params_to_file
import math
import time


def read_dataset_from_h5(h5_file_path, mode):
    dataset = h5py.File(h5_file_path, 'r')
    if mode == 'train':
        train_x = np.array(dataset['train_x'])/255
        train_y = np.array(dataset['train_y'])
        return train_x, train_y
    elif mode == 'test':
        test_x = np.array(dataset['test_x'])/255
        test_y = np.array(dataset['test_y'])
        return test_x, test_y
    elif mode == 'read_params':
        L = len(dataset.keys())//2
        parameters = {}
        for i in range(L):
            parameters['W'+str(i+1)] = dataset['W'+str(i+1)]
            parameters['b'+str(i+1)] = dataset['b'+str(i+1)]
        return parameters


def sigmoid(Z):
    return 1/(1+np.exp(-Z)), Z


def relu(Z):
    return np.maximum(0, Z), Z


def initialize_parameters(layers_dims):
    np.random.seed(42)
    parameters = {}

    for i in range(1, len(layers_dims)):
        parameters['W'+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1])*math.sqrt(2/layers_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A)+b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def compute_cost(A, Y):
    m = Y.shape[1]

    cost = (-1/m)*np.sum(Y * np.log(A) + (1-Y) * np.log(1-A))
    cost = np.squeeze(cost)
    return cost


def sigmoid_backward(dA, cache):
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def update_parameters(parameters, grads, learning_rate):
    num_of_layers = len(parameters)//2

    for i in range(num_of_layers):
        parameters['W'+str(i+1)] = parameters['W'+str(i+1)] - learning_rate*grads['dW'+str(i+1)]
        parameters['b'+str(i+1)] = parameters['b'+str(i+1)] - learning_rate*grads['db'+str(i+1)]
    return parameters


def L_model_forward(X, parameters):
    caches = []
    L = len(parameters)//2
    A = X

    for l in range(1, L+1):
        A_prev = A
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]

        activation = 'relu'
        if l == L:
            activation = 'sigmoid'

        A, cache = linear_activation_forward(A_prev, W, b, activation)
        caches.append(cache)

    return A, caches


def L_model_backward(A, Y, caches):
    dA = -(np.divide(Y, A) - np.divide(1-Y, 1-A))
    L = len(caches)
    grads = {'dA' + str(L+1): dA}

    for l in range(L, 0, -1):
        activation = 'relu'
        if l == L:
            activation = 'sigmoid'
        dA, dW, db = linear_activation_backward(grads['dA'+str(l+1)], caches[l-1], activation)
        grads['dA'+str(l)] = dA
        grads['dW'+str(l)] = dW
        grads['db'+str(l)] = db

    return grads


def predict(X, parameters):
    try:
        m = X.shape[1]
    except IndexError:
        m = 1
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, cathes = L_model_forward(X, parameters)

    for i in range(probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


def deep_model(X, Y, learning_rate, layers_dims, num_iterations=3000, print_cost=False):
    params = initialize_parameters(layers_dims)
    for i in range(num_iterations):
        A, caches = L_model_forward(X, params)

        grads = L_model_backward(A, Y, caches)

        cost = compute_cost(A, Y)

        params = update_parameters(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

    return params


if __name__ == '__main__':
    train_x, train_y = read_dataset_from_h5(r'E:/Peter/data.h5', 'train')
    test_x, test_y = read_dataset_from_h5(r'E:/Peter/data.h5', 'test')
    layers_dims = (train_x.shape[1], 800, 10, 1)
    parameters = deep_model(train_x.T, train_y.T, 0.01, layers_dims, num_iterations=1000, print_cost=True)
    train_predicted = predict(train_x.T, parameters)
    test_predicted = predict(test_x.T, parameters)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_predicted - train_y.T) * 100)))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_predicted - test_y.T) * 100)))
    save_params_to_file(parameters, r'E:/Peter/params.h5')

# ~7h 45min for 10000 iterations with 1850 images
