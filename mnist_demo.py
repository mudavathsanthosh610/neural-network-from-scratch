import os
import argparse
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from nn.network import Network
from nn.layer import Dense
from nn.activations import relu, softmax
from nn.losses import CrossEntropyLoss
from nn.optimizers import SGD, MomentumSGD, Adam
from nn.gradcheck import check_gradients
from nn.visualizer import generate_network_graph

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_PATH = "mnist.npz"

def download_mnist():
    if not os.path.exists(MNIST_PATH):
        print("Downloading MNIST dataset…")
        urllib.request.urlretrieve(MNIST_URL, MNIST_PATH)
        print("Download complete.")
    else:
        print("MNIST dataset already present.")

def load_mnist():
    with np.load(MNIST_PATH) as data:
        X_train, y_train = data["x_train"], data["y_train"]
        X_test, y_test = data["x_test"], data["y_test"]
    # Flatten and normalize to [0, 1]
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0
    # One‑hot encode labels for cross‑entropy loss
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    return X_train, y_train_onehot, X_test, y_test_onehot, y_test

def build_mnist_network(hidden_units: int = 128, activation=relu, output_activation=softmax):
    # Input dim = 784 (28*28), output dim = 10
    net = Network(
        layer_sizes=[784, hidden_units, 10],
        activation=activation,
        loss_fn=CrossEntropyLoss(),
        optimizer=SGD(lr=0.01),  # placeholder, will be replaced later
        batch_size=64,
        shuffle=True,
    )
    # Replace the automatically created layers with desired activations per layer
    # First hidden layer -> ReLU, output layer -> Softmax
    net.layers[0] = Dense(784, hidden_units, activation=activation)
    net.layers[1] = Dense(hidden_units, 10, activation=output_activation)
    return net

def plot_metrics(train_losses, test_accuracies, out_dir="."):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, label="Test accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test accuracy")
    plt.legend()
    loss_path = os.path.join(out_dir, "mnist_loss_accuracy.png")
    plt.tight_layout()
    plt.savefig(loss_path)
    print(f"Saved loss/accuracy plot to {loss_path}")
    return loss_path

def plot_sample_predictions(model: Network, X: np.ndarray, y: np.ndarray, out_dir="."):
    # Take first 25 test images
    samples = X[:25]
    true_labels = np.argmax(y[:25], axis=1)
    preds = model.forward(samples)
    pred_labels = np.argmax(preds, axis=1)
    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(samples[i].reshape(28, 28), cmap="gray")
        plt.title(f"{pred_labels[i]}")
        plt.axis('off')
    sample_path = os.path.join(out_dir, "mnist_samples.png")
    plt.tight_layout()
    plt.savefig(sample_path)
    print(f"Saved sample predictions to {sample_path}")
    return sample_path

def evaluate_accuracy(model: Network, X: np.ndarray, y_onehot: np.ndarray) -> float:
    outputs = model.forward(X)
    preds = np.argmax(outputs, axis=1)
    true = np.argmax(y_onehot, axis=1)
    return np.mean(preds == true)

def main():
    parser = argparse.ArgumentParser(description="MNIST training demo using the custom NN library")
    parser.add_argument("--hidden", type=int, default=128, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Mini‑batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd", help="Optimizer type")
    parser.add_argument("--gradcheck", action="store_true", help="Run gradient check before training")
    args = parser.parse_args()

    download_mnist()
    X_train, y_train, X_test, y_test_onehot, y_test_labels = load_mnist()

    net = build_mnist_network(hidden_units=args.hidden)
    # Set optimizer according to args
    if args.optimizer == "sgd":
        net.optimizer = SGD(lr=args.lr)
    elif args.optimizer == "momentum":
        net.optimizer = MomentumSGD(lr=args.lr, momentum=0.9)
    else:
        net.optimizer = Adam(lr=args.lr)
    net.batch_size = args.batch
    net.loss_fn = CrossEntropyLoss()

    # Gradient check (optional, small batch)
    if args.gradcheck:
        print("Running gradient check on a small batch…")
        X_small = X_train[:5]
        y_small = y_train[:5]
        ok = check_gradients(net, X_small, y_small, loss_fn=CrossEntropyLoss())
        print(f"Gradient check passed: {ok}")

    # Visualize network architecture
    graph_path = generate_network_graph(net.layers, output_path="network_graph.png")
    print(f"Network graph saved to {graph_path}")

    train_losses = []
    test_accuracies = []
    for epoch in trange(args.epochs, desc="Epochs"):
        net.train(X_train, y_train, epochs=1)  # our train method respects batch_size & shuffle
        loss = net.loss_fn.forward(net.output, y_train)  # loss after last batch
        train_losses.append(loss)
        acc = evaluate_accuracy(net, X_test, y_test_onehot)
        test_accuracies.append(acc)
        print(f"Epoch {epoch+1}/{args.epochs} – loss: {loss:.4f}, test acc: {acc:.4f}")

    plot_metrics(train_losses, test_accuracies)
    plot_sample_predictions(net, X_test, y_test_onehot)

if __name__ == "__main__":
    main()
