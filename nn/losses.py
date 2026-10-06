import numpy as np
from typing import Tuple


class MSELoss:
    """Mean Squared Error loss.

    Provides ``forward`` to compute the scalar loss and ``backward`` to compute
    the gradient of the loss with respect to the predictions.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Return the MSE value.

        Parameters
        ----------
        y_pred:
            Predicted values, shape ``(batch, *)`` or ``(*)``.
        y_true:
            Ground‑truth values, same shape as ``y_pred``.
        """
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Derivative of MSE with respect to ``y_pred``.

        Returns ``-2 * (y_true - y_pred) / N`` where ``N`` is the number of
        elements, matching the gradient used in the original script.
        """
        N = y_pred.size
        return -2 * (y_true - y_pred) / N


class CrossEntropyLoss:
    """Cross‑entropy loss for binary or multi‑class classification.

    The implementation assumes probabilities produced by a softmax (or sigmoid
    for binary). ``y_true`` should be provided as one‑hot encoded vectors for the
    multi‑class case or as binary scalars for the binary case.
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        # Binary case – treat shape (N, 1) as scalars
        if y_pred_clipped.ndim == 1 or y_pred_clipped.shape[1] == 1:
            return -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        # Multi‑class case (one‑hot)
        return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        if y_pred_clipped.ndim == 1 or y_pred_clipped.shape[1] == 1:
            return -(y_true / y_pred_clipped) + ((1 - y_true) / (1 - y_pred_clipped))
        return -(y_true / y_pred_clipped) / y_true.shape[0]
