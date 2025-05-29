"""
data_utils.py

Utilities for generating toy social graphs and propagation instances
suitable for testing the full HIM implementation.
"""

import random
import networkx as nx

random.seed(42)  # For reproducibility

def generate_graph(num_nodes: int = 1000,
                   edge_density: float = 0.01,
                   directed: bool = True) -> nx.DiGraph:
    """Generate a random Erdos–Rényi graph."""
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(num_nodes))
    rng = random.random
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u == v:
                continue
            if rng() < edge_density:
                G.add_edge(u, v)
    return G

def simulate_ic(G: nx.DiGraph,
                seed_set,
                p: float = 0.05):
    """Run a single Independent‑Cascade simulation, returning the
    set of activated edges as a directed graph."""
    active = set(seed_set)
    frontier = list(seed_set)
    act_edges = []
    rng = random.random
    while frontier:
        nxt = []
        for u in frontier:
            for v in G.successors(u):
                if v in active:
                    continue
                if rng() < p:
                    active.add(v)
                    act_edges.append((u, v))
                    nxt.append(v)
        frontier = nxt
    H = nx.DiGraph()
    H.add_nodes_from(active)
    H.add_edges_from(act_edges)
    return H

def generate_propagations(G: nx.DiGraph,
                          seed_count: int,
                          num_instances: int = 30,
                          ic_prob: float = 0.05):
    """Return a list of propagation graphs simulated under IC."""
    nodes = list(G.nodes)
    instances = []
    for _ in range(num_instances):
        seeds = random.sample(nodes, seed_count)
        instances.append(simulate_ic(G, seeds, p=ic_prob))
    return instances