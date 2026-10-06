import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def deriv_relu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def deriv_tanh(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def softmax(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

ACTIVATIONS = {
     sigmoid: (sigmoid, deriv_sigmoid),
    relu: (relu, deriv_relu),
    tanh: (tanh, deriv_tanh),
    softmax: (softmax, None),
}
