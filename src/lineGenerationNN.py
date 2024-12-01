import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('../data/historic_clean.csv')

# Split dataset (80/20 split and shuffle)
from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1).values  # Replace 'target' with your actual target column name
y = df['target'].values.reshape(-1, 1)  # Ensure y is a column vector

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize parameters
def initialize_params(layers):
    np.random.seed(1)
    params = {}
    for i in range(1, len(layers)):
        params[f"W{i}"] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
        params[f"b{i}"] = np.zeros((layers[i], 1))
    return params

# ReLU activation function and its derivative
def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_derivative(dA, Z):
    dZ = dA * (Z > 0)
    return dZ

# Forward propagation
def forward_propagation(X, params):
    cache = {"A0": X.T}
    L = len(params) // 2  # Number of layers
    for l in range(1, L + 1):
        W, b = params[f"W{l}"], params[f"b{l}"]
        Z = np.dot(W, cache[f"A{l - 1}"]) + b
        if l == L:  # Output layer
            A = Z  # Linear activation for regression
        else:
            A, _ = relu(Z)  # ReLU for hidden layers
        cache[f"Z{l}"] = Z
        cache[f"A{l}"] = A
    return A, cache

# Cost function (Mean Squared Error for regression)
def compute_cost(y_pred, y):
    m = y.shape[0]
    cost = (1 / (2 * m)) * np.sum((y_pred - y.T) ** 2)
    return cost

# Backward propagation
def backward_propagation(X, y, params, cache):
    grads = {}
    m = X.shape[0]
    L = len(params) // 2  # Number of layers
    y = y.T  # Transpose for consistency

    # Initial gradient (output layer)
    dA = cache[f"A{L}"] - y

    for l in reversed(range(1, L + 1)):
        dZ = dA if l == L else relu_derivative(dA, cache[f"Z{l}"])
        dW = (1 / m) * np.dot(dZ, cache[f"A{l - 1}"].T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        grads[f"dW{l}"] = dW
        grads[f"db{l}"] = db
        if l > 1:
            dA = np.dot(params[f"W{l}"].T, dZ)
    return grads

# Update parameters
def update_params(params, grads, learning_rate):
    L = len(params) // 2  # Number of layers
    for l in range(1, L + 1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return params

# Training function
def train(X, y, layers, learning_rate=0.01, epochs=1000):
    params = initialize_params(layers)
    for epoch in range(epochs):
        y_pred, cache = forward_propagation(X, params)
        cost = compute_cost(y_pred, y)
        grads = backward_propagation(X, y, params, cache)
        params = update_params(params, grads, learning_rate)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.4f}")
    return params

# Define layers (example: input size, hidden layers, output size)
layers = [X_train.shape[1], 10, 1]

# Train the network
trained_params = train(X_train, y_train, layers, learning_rate=0.01, epochs=1000)
