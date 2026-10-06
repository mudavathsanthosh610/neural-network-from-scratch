import numpy as np
from typing import Callable, Optional


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.

    Args:
        x: Input array.
    Returns:
        Array with element‑wise sigmoid applied.
    """
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function w.r.t. its input.

    Args:
        x: Input array (pre‑activation values).
    Returns:
        Derivative of sigmoid evaluated at ``x``.
    """
    s = sigmoid(x)
    return s * (1 - s)


class Dense:
    """Fully‑connected (dense) layer.

    The layer stores its weight matrix ``W`` and bias vector ``b``.
    It supports a forward pass returning the activation and a backward
    pass that computes gradients w.r.t. inputs, weights and biases.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Callable[[np.ndarray], np.ndarray] = sigmoid,
        activation_derivative: Callable[[np.ndarray], np.ndarray] = deriv_sigmoid,
        weight_init: Optional[Callable[[int, int], np.ndarray]] = None,
        bias_init: Optional[Callable[[int], np.ndarray]] = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.activation_derivative = activation_derivative
        # Xavier/Glorot uniform if not supplied
        if weight_init is None:
            limit = np.sqrt(6 / (in_features + out_features))
            self.W = np.random.uniform(-limit, limit, (out_features, in_features))
        else:
            self.W = weight_init(out_features, in_features)
        if bias_init is None:
            self.b = np.zeros(out_features)
        else:
            self.b = bias_init(out_features)
        # Gradients (filled during back‑prop)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Cached values for back‑prop
        self.last_input: Optional[np.ndarray] = None
        self.last_z: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute ``activation(W·x + b)``.

        Args:
            x: Input of shape ``(in_features,)`` or ``(batch, in_features)``.
        Returns:
            Activated output.
        """
        self.last_input = x
        self.last_z = x @ self.W.T + self.b  # shape (batch, out_features) or (out_features,)
        return self.activation(self.last_z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Back‑propagate gradients.

        Args:
            grad_output: Gradient of the loss w.r.t. the layer's output
                (same shape as the forward output).
        Returns:
            Gradient w.r.t. the layer's input.
        """
        # derivative of activation w.r.t. pre‑activation
        delta = grad_output * self.activation_derivative(self.last_z)
        # Gradients w.r.t. parameters
        # If input is a batch, we sum over the batch dimension
        if self.last_input.ndim == 2:
            self.dW = delta.T @ self.last_input
            self.db = delta.sum(axis=0)
        else:
            self.dW = np.outer(delta, self.last_input)
            self.db = delta
        # Gradient w.r.t. input
        grad_input = delta @ self.W
        return grad_input

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """Simple SGD update.

        Args:
            lr: Learning rate.
            weight_decay: L2 regularisation coefficient (added to ``dW``).
        """
        if weight_decay:
            self.dW += weight_decay * self.W
        self.W -= lr * self.dW
        self.b -= lr * self.db

    # Optional helper for inspection
    def __repr__(self) -> str:
        return f"Dense(in={self.in_features}, out={self.out_features})"
