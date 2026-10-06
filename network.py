import numpy as np
from typing import Tuple

# Import the new lightweight deep‑learning library
from nn.network import Network


def main() -> None:
    """Run the original 2‑2‑1 demonstration using the refactored library.

    The dataset and training hyper‑parameters match the original script so the
    printed predictions should be comparable to the previous implementation.
    """
    # Dataset – identical to the original example
    data = np.array([
        [-2, -1],   # Alice
        [25, 6],    # Bob
        [17, 4],    # Charlie
        [-15, -6],  # Diana
    ])
    labels = np.array([1, 0, 0, 1])

    # Build a 2‑2‑1 network (input=2, hidden=2, output=1) using sigmoid
    net = Network([2, 2, 1])

    # Train – same hyper‑parameters as the original code
    loss_history = net.train(data, labels, epochs=1000, lr=0.1)
    # Print loss every 10 epochs (the train method records them)
    for i, loss in enumerate(loss_history):
        epoch = i * 10
        print(f"Epoch {epoch:4d} loss: {loss:.3f}")

    # Final predictions for two example points (extract scalar with .item())
    emily = np.array([-7, -3])
    frank = np.array([20, 2])
    emily_pred = net.forward(emily).item()
    frank_pred = net.forward(frank).item()
    print(f"Emily: {emily_pred:.3f} (expected ~= 0.95)")
    print(f"Frank: {frank_pred:.3f} (expected ~= 0.04)")


if __name__ == "__main__":
    main()
