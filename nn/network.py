import numpy as np
from typing import List, Callable, Optional

from .layer import Dense
from .losses import MSELoss, CrossEntropyLoss
from .optimizers import SGD, Optimizer


class Network:
    """Simple feed‑forward neural network composed of ``Dense`` layers.

    The architecture is defined by a list of layer sizes, e.g. ``[2, 8, 4, 1]``
    creates a network with input dimension 2, two hidden layers of sizes 8 and 4,
    and a single output neuron. All hidden layers use the same activation (by
    default sigmoid) but this can be overridden per layer.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activation: Optional[Callable] = None,
        activation_derivative: Optional[Callable] = None,
        loss_fn: Optional[object] = None,
        optimizer: Optional[Optimizer] = None,
        batch_size: int = 1,
        shuffle: bool = True,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers")
        # Default to sigmoid if no activation supplied
        from .layer import sigmoid, deriv_sigmoid
        if activation is None:
            activation = sigmoid
        if activation_derivative is None:
            activation_derivative = deriv_sigmoid
        # Build layers (no activation on the final layer by default)
        self.layers: List[Dense] = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            act = activation if out_dim != 1 else activation  # keep activation on output for now
            self.layers.append(
                Dense(
                    in_features=in_dim,
                    out_features=out_dim,
                    activation=act,
                    activation_derivative=activation_derivative,
                )
            )
        # Training configuration
        self.loss_fn = loss_fn if loss_fn is not None else MSELoss()
        self.optimizer = optimizer if optimizer is not None else SGD(lr=0.1)  # default lr; user can override via optimizer
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Propagate *x* through all layers and return the final output.
        Supports a 1‑D input vector or a 2‑D batch (``batch, features``).
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: np.ndarray) -> None:
        """Back‑propagate the gradient of the loss w.r.t. the network output.

        Parameters
        ----------
        loss_grad:
            Gradient of the loss with respect to the network output.
        """
        """Back‑propagate the gradient of the loss w.r.t. the network output.
        ``loss_grad`` should have the same shape as the output of ``forward``.
        """
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self) -> None:
        """Update all parameters using the configured optimizer."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set for the network.")
        self.optimizer.step(self.layers)

        """Update all parameters using the configured optimizer.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not set for the network.")
        self.optimizer.step(self.layers)

    # Helper for training the classic 2‑2‑1 demo (supports mini‑batch & shuffling)
    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1000,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> List[float]:
        """Train the network.

        Parameters
        ----------
        data, labels:
            Training dataset.
        epochs:
            Number of passes over the full dataset.
        lr:
            Learning rate – overrides optimizer's lr if provided.
        weight_decay:
            L2 regularisation coefficient passed to the optimizer (if supported).

        Returns
        -------
        List[float]
            Recorded loss values every 10 epochs.
        """
        if lr is not None and hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr
        loss_history: List[float] = []
        num_samples = data.shape[0]
        for epoch in range(epochs):
            if self.shuffle:
                indices = np.random.permutation(num_samples)
                data_shuffled = data[indices]
                labels_shuffled = labels[indices]
            else:
                data_shuffled = data
                labels_shuffled = labels
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_x = data_shuffled[start:end]
                batch_y = labels_shuffled[start:end]
                preds = self.forward(batch_x)
                loss_val = self.loss_fn.forward(preds, batch_y)
                grads = self.loss_fn.backward(preds, batch_y)
                self.backward(grads)
                self.step()
            if epoch % 10 == 0:
                total_loss = self.loss_fn.forward(self.forward(data), labels)
                loss_history.append(total_loss)
        return loss_history

    def gradient_check(self, x: np.ndarray, y: np.ndarray, epsilon: float = 1e-5, tolerance: float = 1e-4) -> bool:
        """Numerically verify the gradients computed by back‑propagation.

        Parameters
        ----------
        x, y:
            Single input‑output pair (or batch) to use for the check.
        epsilon:
            Small perturbation for finite‑difference approximation.
        tolerance:
            Maximum acceptable relative error.
        """
        # Forward pass and analytical gradients
        pred = self.forward(x)
        loss = self.loss_fn.forward(pred, y)
        grad_loss = self.loss_fn.backward(pred, y)
        self.backward(grad_loss)

        max_rel_error = 0.0
        # Check each layer's parameters
        for layer in self.layers:
            # Weight gradients
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    original = layer.W[i, j]
                    layer.W[i, j] = original + epsilon
                    loss_plus = self.loss_fn.forward(self.forward(x), y)
                    layer.W[i, j] = original - epsilon
                    loss_minus = self.loss_fn.forward(self.forward(x), y)
                    layer.W[i, j] = original
                    num_grad = (loss_plus - loss_minus) / (2 * epsilon)
                    ana_grad = layer.dW[i, j]
                    rel_error = abs(num_grad - ana_grad) / max(abs(num_grad), abs(ana_grad), 1e-8)
                    max_rel_error = max(max_rel_error, rel_error)
            # Bias gradients
            for i in range(layer.b.shape[0]):
                original = layer.b[i]
                layer.b[i] = original + epsilon
                loss_plus = self.loss_fn.forward(self.forward(x), y)
                layer.b[i] = original - epsilon
                loss_minus = self.loss_fn.forward(self.forward(x), y)
                layer.b[i] = original
                num_grad = (loss_plus - loss_minus) / (2 * epsilon)
                ana_grad = layer.db[i]
                rel_error = abs(num_grad - ana_grad) / max(abs(num_grad), abs(ana_grad), 1e-8)
                max_rel_error = max(max_rel_error, rel_error)
        print(f"Max relative gradient error: {max_rel_error:.2e}")
        return max_rel_error < tolerance

    def __repr__(self) -> str:
        layer_desc = " -> ".join(str(l) for l in self.layers)
        return f"Network({layer_desc})"
    def train(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 1000,
        lr: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> List[float]:
        """Train the network.

        Parameters
        ----------
        data, labels:
            Training dataset.
        epochs:
            Number of passes over the full dataset.
        lr:
            Learning rate – overrides optimizer's lr if provided.
        weight_decay:
            L2 regularisation coefficient passed to the optimizer (if supported).

        Returns
        -------
        List[float]
            Recorded loss values every 10 epochs.
        """
        if lr is not None and hasattr(self.optimizer, "lr"):
            self.optimizer.lr = lr
        loss_history: List[float] = []
        num_samples = data.shape[0]
        for epoch in range(epochs):
            # Optional shuffling
            if self.shuffle:
                indices = np.random.permutation(num_samples)
                data_shuffled = data[indices]
                labels_shuffled = labels[indices]
            else:
                data_shuffled = data
                labels_shuffled = labels
            # Mini‑batch loop
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_x = data_shuffled[start:end]
                batch_y = labels_shuffled[start:end]
                # Forward pass for the batch
                preds = self.forward(batch_x)
                # Compute loss and gradient
                loss_val = self.loss_fn.forward(preds, batch_y)
                grads = self.loss_fn.backward(preds, batch_y)
                # Back‑propagation
                self.backward(grads)
                # Parameter update
                self.step()
            if epoch % 10 == 0:
                # Full‑dataset loss for monitoring
                total_loss = self.loss_fn.forward(self.forward(data), labels)
                loss_history.append(total_loss)
        return loss_history

    def __repr__(self) -> str:
        layer_desc = " -> ".join(str(l) for l in self.layers)
        return f"Network({layer_desc})"
