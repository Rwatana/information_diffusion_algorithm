"""
data_utils.py

Utilities for generating toy social graphs and propagation instances.
The generate_graph function now creates weighted directed graphs by default.
"""

import random
import networkx as nx

random.seed(42)  # For reproducibility


def generate_graph(num_nodes: int = 100,
                   edge_density: float = 0.05,
                   directed: bool = True,
                   min_weight: float = 0.01,
                   max_weight: float = 0.2) -> nx.DiGraph:
    """
    Generate a random Erdos–Rényi graph.
    By default, it's a directed graph where each edge has a 'weight' attribute
    representing a propagation probability, randomly chosen between min_weight and max_weight.
    """
    if not directed:
        # If a non-directed graph is explicitly requested, create it without weights
        # or adapt to add weights symmetrically if needed (current implementation doesn't).
        # For this request, we focus on the default weighted directed graph.
        G_undirected = nx.Graph()
        G_undirected.add_nodes_from(range(num_nodes))
        rng = random.random
        # For undirected, only consider (u,v) where u < v to avoid duplicate checks
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if rng() < edge_density:
                    G_undirected.add_edge(u, v)
        return G_undirected # Returns an unweighted undirected graph if directed=False

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    rng = random.random
    
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u == v:  # No self-loops
                continue
            
            # Consider edge u -> v
            if rng() < edge_density:
                weight_uv = random.uniform(min_weight, max_weight)
                G.add_edge(u, v, weight=weight_uv)
            
            # Note: The loop structure naturally handles the potential for an edge v -> u.
            # When the outer loop reaches 'v' and inner loop reaches 'u',
            # the edge v -> u will be considered independently.
            # This allows for:
            # 1. No edge between u and v
            # 2. Edge u -> v only (with its own weight)
            # 3. Edge v -> u only (with its own weight)
            # 4. Edges u -> v AND v -> u (each with its own independent weight)

    return G


def simulate_ic(G: nx.DiGraph,
                seed_set: set, # Changed from just 'seed_set' to 'seed_set: set' for type hinting
                default_activation_prob: float = 0.1): # Retained for fallback
    """
    Run a single Independent‑Cascade simulation.
    Uses 'weight' attribute on edges for propagation probability if available,
    otherwise uses default_activation_prob.
    Returns the set of activated nodes and a log of activated edges.
    """
    active_nodes = set(seed_set)
    newly_activated_in_current_step = set(seed_set) 
    activated_edges_log = [] 
    step = 0

    while newly_activated_in_current_step:
        current_frontier = list(newly_activated_in_current_step) 
        newly_activated_in_this_iteration = set() 
        step += 1

        for u_node in current_frontier: # Renamed u to u_node to avoid conflict with outer scope u
            if u_node not in G: 
                continue
            for v_node in G.successors(u_node): # Renamed v to v_node
                if v_node not in active_nodes: 
                    # Get propagation probability: from edge 'weight' or use default
                    prob = G[u_node][v_node].get('weight', default_activation_prob)
                    
                    if random.random() < prob: 
                        active_nodes.add(v_node) 
                        newly_activated_in_this_iteration.add(v_node) 
                        activated_edges_log.append({
                            "source": u_node,
                            "target": v_node,
                            "step": step, 
                            "prob_used": prob
                        })
        
        newly_activated_in_current_step = newly_activated_in_this_iteration
        
    return active_nodes, activated_edges_log

def generate_propagations(G: nx.DiGraph,
                          seed_count: int,
                          num_instances: int = 30,
                          # ic_prob is now effectively the default_activation_prob for simulate_ic
                          # if edges in G don't have weights (which they should with the new generate_graph)
                          default_ic_prob: float = 0.05): 
    """
    Return a list of (activated_nodes_set, activated_edges_log) tuples 
    simulated under IC.
    """
    if not G.nodes: # Check if the graph has nodes
        return []
        
    nodes = list(G.nodes)
    instances = []
    for _ in range(num_instances):
        # Ensure seed_count is not greater than the number of available nodes
        actual_seed_count = min(seed_count, len(nodes))
        if actual_seed_count == 0: # If no nodes to sample from, or seed_count is 0
            seeds = set()
        else:
            seeds = random.sample(nodes, actual_seed_count)
        
        # simulate_ic returns a tuple (activated_nodes_set, activated_edges_log)
        # The old code was instances.append(simulate_ic(G, seeds, p=ic_prob))
        # which would have made instances a list of tuples.
        # The HIM model's fit method likely expects propagation graphs (nx.DiGraph objects)
        # or at least a list of activated edges to construct them.
        # For now, I'll keep it returning what simulate_ic returns.
        # If HIM needs DiGraph objects, this part needs further adjustment.
        activated_nodes_set, activated_edges_log = simulate_ic(G, set(seeds), default_activation_prob=default_ic_prob)
        
        # Storing the log of activated edges might be more useful for HIM than the graph object itself,
        # or the HIM model might expect graph objects.
        # For now, let's return the log which is more flexible.
        # If graph objects are needed, they can be constructed from these logs.
        # For consistency with what HIM might expect (propagation graphs),
        # let's return graph objects created from the activated edges.
        
        H_prop = nx.DiGraph()
        if activated_nodes_set: # Add nodes that were part of the cascade
            H_prop.add_nodes_from(activated_nodes_set)
        if activated_edges_log: # Add edges that were part of the cascade
            for edge_info in activated_edges_log:
                H_prop.add_edge(edge_info["source"], edge_info["target"], weight=edge_info["prob_used"])
        instances.append(H_prop)

    return instances
