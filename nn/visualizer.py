import os
from typing import List
import graphviz

def generate_network_graph(layers: List[object], output_path: str = "network.png") -> str:
    """Generate a simple directed graph visualizing the network architecture.

    Args:
        layers: List of layer objects (expects ``Dense`` instances with ``input_dim``
            and ``output_dim`` attributes).
        output_path: Path where the rendered image will be saved (PNG format).

    Returns:
        The absolute path to the generated image.
    """
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="LR")
    dot.attr("node", shape="record")
    for idx, layer in enumerate(layers):
        name = f"Layer{idx}\n{type(layer).__name__}\n{layer.input_dim}->{layer.output_dim}"
        dot.node(str(idx), name)
        if idx > 0:
            dot.edge(str(idx - 1), str(idx))
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dot.render(filename=output_path, cleanup=True)
    return os.path.abspath(output_path + ".png")
