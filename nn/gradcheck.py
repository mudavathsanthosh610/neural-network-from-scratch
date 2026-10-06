import numpy as np
from typing import Callable, Tuple

def _flatten_params(layers):
    """Return a list of (param, grad) tuples for all layers."""
    params = []
    for layer in layers:
        if hasattr(layer, 'weight'):
            params.append((layer.weight, layer.dweight))
        if hasattr(layer, 'bias'):
            params.append((layer.bias, layer.dbias))
    return params

def numeric_gradient(network, X: np.ndarray, y: np.ndarray, loss_fn, epsilon: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """Compute numeric gradients for all parameters in the network.

    Returns a tuple (num_grads_weights, num_grads_biases) matching the shape of the
    analytic gradients stored after a backward pass.
    """
    # Ensure forward/backward have been run to get shapes
    network.forward(X)
    loss = loss_fn.forward(network.output, y)
    loss_fn.backward()
    # Backward to get analytic grads
    network.backward(loss_fn.doutput)

    num_grads_w = []
    num_grads_b = []
    for layer in network.layers:
        if not hasattr(layer, 'weight'):
            continue
        w_shape = layer.weight.shape
        b_shape = layer.bias.shape
        w_num = np.zeros_like(layer.weight)
        b_num = np.zeros_like(layer.bias)
        # gradient for each weight element
        it = np.nditer(layer.weight, flags=['multi_index'], op_flags=['readwrite'])
        for _ in it:
            idx = it.multi_index
            orig = layer.weight[idx]
            layer.weight[idx] = orig + epsilon
            loss_plus = loss_fn.forward(network.forward(X), y)
            layer.weight[idx] = orig - epsilon
            loss_minus = loss_fn.forward(network.forward(X), y)
            layer.weight[idx] = orig
            w_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        # bias gradients
        itb = np.nditer(layer.bias, flags=['multi_index'], op_flags=['readwrite'])
        for _ in itb:
            idx = itb.multi_index
            orig = layer.bias[idx]
            layer.bias[idx] = orig + epsilon
            loss_plus = loss_fn.forward(network.forward(X), y)
            layer.bias[idx] = orig - epsilon
            loss_minus = loss_fn.forward(network.forward(X), y)
            layer.bias[idx] = orig
            b_num[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        num_grads_w.append(w_num)
        num_grads_b.append(b_num)
    return num_grads_w, num_grads_b

def check_gradients(network, X: np.ndarray, y: np.ndarray, loss_fn=None, epsilon: float = 1e-5, tolerance: float = 1e-4) -> bool:
    """Perform gradient checking.

    Returns ``True`` if the maximum absolute difference between analytic and numeric
    gradients is below ``tolerance``.
    """
    if loss_fn is None:
        from .losses import MSELoss
        loss_fn = MSELoss()
    # Run forward and backward to compute analytic grads
    network.forward(X)
    loss = loss_fn.forward(network.output, y)
    loss_fn.backward()
    network.backward(loss_fn.doutput)

    num_w, num_b = numeric_gradient(network, X, y, loss_fn, epsilon)
    # Compare
    max_diff = 0.0
    for layer, w_num, b_num in zip(network.layers, num_w, num_b):
        if hasattr(layer, 'dweight'):
            diff_w = np.max(np.abs(layer.dweight - w_num))
            max_diff = max(max_diff, diff_w)
        if hasattr(layer, 'dbias'):
            diff_b = np.max(np.abs(layer.dbias - b_num))
            max_diff = max(max_diff, diff_b)
    print(f"Maximum gradient difference: {max_diff}")
    return max_diff < tolerance
