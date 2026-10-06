import os
from flask import Flask, request, jsonify, send_file
import numpy as np

# Ensure the repository root is on the Python path for imports
import sys
repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from nn.network import Network
from nn.layer import Dense
from nn.gradcheck import check_gradients
from nn.visualizer import generate_network_graph

app = Flask(__name__, static_folder=repo_root, static_url_path="")

# ----------------------------------------------------
# Gradient check endpoint
# ----------------------------------------------------
@app.route('/gradcheck', methods=['POST'])
def gradcheck_endpoint():
    data = request.get_json()
    if not data or 'X' not in data or 'y' not in data:
        return jsonify({'error': 'Missing X or y in payload'}), 400
    X = np.array(data['X'], dtype=float)
    y = np.array(data['y'], dtype=float)
    # Build a tiny network matching the shape of the data
    input_dim = X.shape[1]
    net = Network(layer_sizes=[input_dim, 4, 1], activation='relu')
    # Use the same loss function as the demo (MSE)
    from nn.losses import MSELoss
    loss_fn = MSELoss()
    # Perform gradient check
    ok = check_gradients(net, X, y, loss_fn=loss_fn)
    return jsonify({'ok': ok, 'message': 'Gradient check completed'})

# ----------------------------------------------------
# Network graph endpoint – generates a PNG visualisation
# ----------------------------------------------------
@app.route('/graph', methods=['GET'])
def graph_endpoint():
    # Create a simple example network (you can replace this with the
    # currently trained network if you expose it via a global variable).
    net = Network(layer_sizes=[2, 4, 1], activation='relu')
    # Generate graph image in the repository root (static folder)
    img_path = os.path.abspath(os.path.join(repo_root, 'network_graph.png'))
    generate_network_graph(net.layers, output_path=img_path)
    return send_file(img_path, mimetype='image/png')

    # ----------------------------------------------------
    # MNIST training endpoint – runs mnist_demo with provided hyper‑parameters
    # ----------------------------------------------------
    @app.route('/train', methods=['POST'])
    def train_endpoint():
        # Expected JSON payload with hyper‑parameters
        data = request.get_json() or {}
        # Build command line arguments for mnist_demo.py
        args = []
        hidden = data.get('hidden', 128)
        epochs = data.get('epochs', 5)
        batch = data.get('batch', 64)
        lr = data.get('lr', 0.01)
        optimizer = data.get('optimizer', 'sgd')
        gradcheck = data.get('gradcheck', False)
        args += [f'--hidden={hidden}', f'--epochs={epochs}', f'--batch={batch}', f'--lr={lr}', f'--optimizer={optimizer}']
        if gradcheck:
            args.append('--gradcheck')
        # Run mnist_demo as a subprocess
        import subprocess, shlex, sys
        cmd = [sys.executable, os.path.join(repo_root, 'mnist_demo.py')] + args
        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'Training failed: {e}'}), 500
        # After training, the demo saves plots in the repo root
        # Return paths to the generated images (static folder will serve them)
        return jsonify({
            'message': 'Training completed',
            'loss_accuracy_plot': 'mnist_loss_accuracy.png',
            'sample_predictions_plot': 'mnist_samples.png',
            'network_graph': 'network_graph.png'
        })

    # Run on localhost, port 5000 – you can change as needed
    if __name__ == '__main__':
        print('🚀 Starting Flask server on http://127.0.0.1:5000/')
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
