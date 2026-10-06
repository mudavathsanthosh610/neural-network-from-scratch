# Neural Network From Scratch

An interactive browser demo that teaches the basics of binary classification with a tiny neural network.

The original Python example was written for [Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/).

## Interactive browser demo

Open `index.html` directly in a modern browser, or serve the project locally:

```bash
python -m http.server 8000
```

Then visit https://mudavathsanthosh610.github.io/neural-network-from-scratch/ The browser demo runs without a backend and includes:

- Live training loss and a progress indicator
- Editable two-number inputs and training settings
- A prediction and probability for Class 1
- A 2D decision boundary and draggable 3D probability surface

The example is for learning and visualization. It uses four toy training points and is not intended for reliable real-world predictions.

## Python demo

To run the Python example, install [NumPy](https://numpy.org/) if needed:

```bash
python -m pip install numpy
```

Then run:

```bash
python network.py
```

## Deployment

The browser demo is a static site and can be deployed with GitHub Pages by selecting the `main` branch and `/ (root)` as the publishing source in the repository's Pages settings.
