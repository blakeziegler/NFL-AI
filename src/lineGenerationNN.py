# Line Generation Neural Network

import pandas as pd
import numpy as np

df = pd.read_csv('../data/historic_clean.csv')



'''
        NEURAL NETWORK SETUP
- Split (80/20) and shuffle data
- Define layer dimensions
- initialize parameters
'''

class NeuralNetwork:
    # Initialize getters
    def __init__(self, layers, activation='relu', optimizer='momentum', learning=0.01, beta=0.9):
        self.layers = layers
        self.activation = activation
        self.optimizer = optimizer
        self.learning = learning
        self.beta = beta
        self.params = self._init_params()
        self.velo = self._init_velo()
        self.cache = {}
        self.gradient = {}

    # Set activation function
    def _activation_function(self, x):
        if x == 'relu':
            return self._relu, self._relu_derivative
        else:
            raise ValueError()

    # initialize parameters, weights, and bias
    def _init_params(self):
        np.random.seed(0)
        params = {}
        L = len(self.layers)
        # He Initialization Formula: W^[layer] ~ N(0, (2/n^(layer-1))
        for layer in range(1, L):
            params['W' + str(layer)] = \
                (np.random.randn(self.layers[layer], self.layers[layer - 1])
                 ) * np.sqrt(2 / self.layers(layer - 1))
            params['b' + str(layer)] = np.zeros((self.layers[layer], 1))
            return params

    # Initialize velocity for momentum calculation
    def _init_velo(self):
        velo = {}
        L = len(self.layers)
        for layer in range(1, L):
            velo['dW'+ str(layer)] = np.zeros_like(self.params['W' + str(layer)])
            velo['db'+ str(layer)] = np.zeros_like(self.params['b' + str(layer)])
            print(velo)
        return velo


    '''
        ACTIVATION FUNCTIONS
    - Define linear and relu algorithms along w/ their derivatives
    '''

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self,Z ):
        return (Z > 0).astype(float)

    def _linear(self,Z ):
        return Z

    
    " FORWARD PROP- Define input -> output algorithm- compute cost function"
    
    def forward_feed(self, X):
        # Set network params
        A = X.T
        self.cache['A'] = A
        L = len(self.layers) - 1
        func, _ = self.activation

        # Hidden Layers
        for layer in range(1, L):
            W = self.params['W' + str(layer)]
            b = self.params['b' + str(layer)]
            Z = np.dot(W, A) + b
            A = func(Z)
            self.cache['A' + str(layer)] = A
            self.cache['Z' + str(layer)] = Z

        # Output Layer
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        Z = np.dot(W, A) + b
        A = self._linear
        self.cache['A' + str(L)] = A
        self.cache['Z' + str(L)] = Z

    def cost(self, y1, y2):
        slope = y2.shape[1]
        C = np.sum((y1 - y2) ** 2 / (2 * slope))
        return C

    '''
        BACK PROP
    - Move back through network driven by cost value
    - Update parameters
    '''

    def backward_feed(self, Y_true):
        L = len(self.layers) - 1
        m = Y_true.shape[1]

        # Output layer gradients
        A = self.cache['A' + str(L)]
        Z = self.cache['Z' + str(L)]
        self.gradient['dZ' + str(L)] = A - Y_true
        self.gradient['dW' + str(L)] = (1 / m) * np.dot(self.gradient['dZ' + str(L)], self.cache['A' + str(L - 1)].T)
        self.gradient['db' + str(L)] = (1 / m) * np.sum(self.gradient['dZ' + str(L)], axis=1, keepdims=True)

        # Hidden layer gradients
        for layer in range(L - 1, 0, -1):
            dZ_next = self.gradient['dZ' + str(layer + 1)]
            W_next = self.params['W' + str(layer + 1)]
            Z = self.cache['Z' + str(layer)]
            self.gradient['dZ' + str(layer)] = np.dot(W_next.T, dZ_next) * self._relu_derivative(Z)
            self.gradient['dW' + str(layer)] = (1 / m) * np.dot(self.gradient['dZ' + str(layer)], self.cache['A' + str(layer - 1)].T)
            self.gradient['db' + str(layer)] = (1 / m) * np.sum(self.gradient['dZ' + str(layer)], axis=1, keepdims=True)


    def update_network(self):
        L = len(self.layers) - 1
        for layer in range(1, L + 1):
            if self.optimizer == 'momentum':
                self.velo['dW' + str(layer)] = self.beta * self.velo['dW' + str(layer)] + (1 - self.beta) * self.gradient['dW' + str(layer)]
                self.velo['db' + str(layer)] = self.beta * self.velo['db' + str(layer)] + (1 - self.beta) * self.gradient['db' + str(layer)]

                self.params['W' + str(layer)] -= self.learning * self.velo['dW' + str(layer)]
                self.params['b' + str(layer)] -= self.learning * self.velo['db' + str(layer)]
            else:  # Basic gradient descent
                self.params['W' + str(layer)] -= self.learning * self.gradient['dW' + str(layer)]
                self.params['b' + str(layer)] -= self.learning * self.gradient['db' + str(layer)]



    
        'TRAINING AND OPTIMIZATION'
    

    def train(self, X, Y, epochs=1000, batchsize=64):
        slope = X.shape[0]
        X_T = X.T
        Y_T = Y.reshape(1, -1)
        batches = slope // batchsize
        cost_vals = []
        for epoch in range(epochs):
            permutate = np.random.permutation(slope)
            X_scramble = X[permutate]
            Y_scramble = Y[permutate]
            for batch in range(batches):
                start = batch * batchsize
                end = min(start + batchsize, slope)
                X_batch = X_scramble[start:end]
                Y_batch = Y_scramble[start:end]
                Y_pred = self.forward_feed(X_batch)
                C = self.cost(Y_pred, Y_batch.T)
                self.backward_feed(Y_batch.T)
                self.update_network()
                print(f"Training Round: {epoch}; Cost: {C}")
            return cost_vals
        def prediction(self, X_pred):
            Y_pred = self.forward_feed(X_pred)
            return Y_pred.T