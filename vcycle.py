import matplotlib.pyplot as plt
import networkx as nx

# Define internal level keys for logic
level_keys = ['L0', 'L1', 'L2', 'L3']

# Define LaTeX labels for display
level_labels = {
    'L0': r'$\Omega^h$',
    'L1': r'$\Omega^{2h}$',
    'L2': r'$\Omega^{4h}$',
    'L3': r'$\Omega^{8h}$'
}

# Alias for clarity
L0, L1, L2, L3 = level_keys

# Define cycles using logical level keys
cycles = {
    "V": [L0, L1, L2, L3, L2, L1, L0],
    "W": [L0, L1, L2, L3, L2, L3, L2, L1, L2, L3, L2, L3, L2, L1, L0],  # Double W shape
    "F": [L0, L1, L2, L3, L2, L3, L2, L1, L2, L3, L2, L1, L0]  # W then V from L1
}

# Colors for node activities
node_colors = {
    'smoothing': 'skyblue',
    'restriction': 'orange',
    'prolongation': 'green',
    'coarse': 'red',
    'default': 'lightgray'
}

edge_colors = {
    'restriction': 'orange',
    'prolongation': 'green',
    'default': 'black'
}

# Highlight specific nodes and edges
highlight_nodes = {
    'L3': 'coarse',
}

highlight_edges = {
    ('L0', 'L1'): 'restriction',
    ('L1', 'L2'): 'restriction',
    ('L2', 'L3'): 'restriction',
    ('L3', 'L2'): 'prolongation',
    ('L2', 'L1'): 'prolongation',
    ('L1', 'L0'): 'prolongation'
}

def plot_cycle(ax, cycle):
    G = nx.DiGraph()
    pos = {}

    for i, level in enumerate(cycle):
        G.add_node(i, label=level)
        pos[i] = (i, -level_keys.index(level))  # Position by level depth
        if i > 0:
            G.add_edge(i - 1, i)

    node_color_list = [node_colors.get(highlight_nodes.get(G.nodes[n]['label'], 'default'), 'lightgray') for n in G.nodes]

    edge_color_list = []
    for u, v in G.edges:
        label_u = G.nodes[u]['label']
        label_v = G.nodes[v]['label']
        color = edge_colors.get(highlight_edges.get((label_u, label_v), 'default'), 'black')
        edge_color_list.append(color)

    labels = {n: level_labels[G.nodes[n]['label']] for n in G.nodes}
    nx.draw(G, pos, ax=ax, with_labels=True, labels=labels,
            node_color=node_color_list, edge_color=edge_color_list,
            arrows=True, node_size=1500, font_size=16, width=2)
    ax.axis('off')

fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 5, 4]})
plot_cycle(axes[0], cycles["V"])
plot_cycle(axes[1], cycles["W"])
plot_cycle(axes[2], cycles["F"])
plt.tight_layout()
plt.savefig("multigrid_cycles.png")