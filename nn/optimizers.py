import numpy as np
from typing import List, Tuple, Optional

# Simple abstract optimizer interface
class Optimizer:
    """Base class for optimizers.
    Sub‑classes must implement ``step(layers)`` which updates the parameters of
    the provided layers in‑place.
    """
    def step(self, layers: List[object]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent.

    Parameters
    ----------
    lr: float
        Learning rate.
    weight_decay: float, default 0.0
        L2 regularisation coefficient (added to the gradient of the weights).
    """
    def __init__(self, lr: float, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers: List[object]) -> None:
        for layer in layers:
            layer.update_params(self.lr, self.weight_decay)


class MomentumSGD(Optimizer):
    """SGD with momentum.

    Parameters
    ----------
    lr: float
        Learning rate.
    momentum: float, default 0.9
        Momentum factor.
    weight_decay: float, default 0.0
        L2 regularisation coefficient.
    """
    def __init__(self, lr: float, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities: List[Tuple[np.ndarray, np.ndarray]] = []  # (v_W, v_b)

    def _ensure_velocities(self, layers: List[object]) -> None:
        if not self.velocities:
            # Initialise velocity buffers with zeros matching each layer's shape
            for layer in layers:
                self.velocities.append((np.zeros_like(layer.W), np.zeros_like(layer.b)))

    def step(self, layers: List[object]) -> None:
        self._ensure_velocities(layers)
        for layer, (v_W, v_b) in zip(layers, self.velocities):
            # Apply L2 regularisation if requested
            if self.weight_decay:
                layer.dW += self.weight_decay * layer.W
            # Update velocities
            v_W[:] = self.momentum * v_W - self.lr * layer.dW
            v_b[:] = self.momentum * v_b - self.lr * layer.db
            # Apply updates
            layer.W += v_W
            layer.b += v_b


class Adam(Optimizer):
    """Adam optimiser.

    Parameters
    ----------
    lr: float, default 0.001
    beta1: float, default 0.9
    beta2: float, default 0.999
    eps: float, default 1e-8
    weight_decay: float, default 0.0
    """
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: List[Tuple[np.ndarray, np.ndarray]] = []  # first moment
        self.v: List[Tuple[np.ndarray, np.ndarray]] = []  # second moment
        self.t = 0

    def _ensure_state(self, layers: List[object]) -> None:
        if not self.m:
            for layer in layers:
                self.m.append((np.zeros_like(layer.W), np.zeros_like(layer.b)))
                self.v.append((np.zeros_like(layer.W), np.zeros_like(layer.b)))

    def step(self, layers: List[object]) -> None:
        self.t += 1
        self._ensure_state(layers)
        for layer, (m_W, m_b), (v_W, v_b) in zip(layers, self.m, self.v):
            # L2 regularisation on gradients
            if self.weight_decay:
                layer.dW += self.weight_decay * layer.W

            # Update biased first moment estimate
            m_W[:] = self.beta1 * m_W + (1 - self.beta1) * layer.dW
            m_b[:] = self.beta1 * m_b + (1 - self.beta1) * layer.db
            # Update biased second raw moment estimate
            v_W[:] = self.beta2 * v_W + (1 - self.beta2) * (layer.dW ** 2)
            v_b[:] = self.beta2 * v_b + (1 - self.beta2) * (layer.db ** 2)

            # Compute bias‑corrected estimates
            m_W_hat = m_W / (1 - self.beta1 ** self.t)
            m_b_hat = m_b / (1 - self.beta1 ** self.t)
            v_W_hat = v_W / (1 - self.beta2 ** self.t)
            v_b_hat = v_b / (1 - self.beta2 ** self.t)

            # Parameter update
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
