import random
import networkx as nx

__all__ = ["generate_graph", "simulate_ic", "generate_propagations"]

# ------------------------------------------------------------------ #
# 1.  Graph generator                                                 #
# ------------------------------------------------------------------ #
def generate_graph(
    num_nodes: int = 100,
    edge_density: float = 0.05,
    directed: bool = True,
    min_weight: float = 0.01,
    max_weight: float = 0.20,
) -> nx.DiGraph | nx.Graph:
    """
    Erdős–Rényi G(n, p) generator.

    * **Every** created edge carries a float weight in [min_weight, max_weight].
    * For directed=True the (u→v) and (v→u) trials are independent, so
      you naturally obtain 0-, 1- or 2-way connections with different weights.
    """
    rng = random.random

    if directed:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for u in range(num_nodes):
            for v in range(num_nodes):
                if u == v:
                    continue                    # no self loops
                if rng() < edge_density:
                    G.add_edge(
                        u,
                        v,
                        weight=random.uniform(min_weight, max_weight),
                    )
        return G

    # undirected variant
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng() < edge_density:
                w = random.uniform(min_weight, max_weight)
                G.add_edge(u, v, weight=w)
    return G


# ------------------------------------------------------------------ #
# 2.  IC (Independent Cascade) simulation                             #
# ------------------------------------------------------------------ #
def simulate_ic(G: nx.DiGraph, seed_set: set[int]):
    """
    One IC rollout.

    Raises KeyError *immediately* if an encountered edge lacks 'weight'
    (this should never happen when the graph is produced by
    `generate_graph`).
    Returns
        activated_nodes: set[int]
        log: List[dict] – one entry per successful activation
    """
    active = set(seed_set)
    frontier = set(seed_set)
    log = []
    step = 0

    while frontier:
        step += 1
        next_frontier = set()
        for u in frontier:
            for v in G.successors(u):
                if v in active:
                    continue
                # Enforce presence of weight attribute
                if "weight" not in G[u][v]:
                    raise KeyError(f"edge ({u}→{v}) missing 'weight'")
                if random.random() < G[u][v]["weight"]:
                    active.add(v)
                    next_frontier.add(v)
                    log.append(
                        dict(source=u, target=v, step=step, prob_used=G[u][v]["weight"])
                    )
        frontier = next_frontier

    return active, log


# ------------------------------------------------------------------ #
# 3.  Convenience: produce many propagation graphs                    #
# ------------------------------------------------------------------ #
def generate_propagations(G: nx.DiGraph, seed_count: int, *, num_instances: int = 30):
    """
    Returns a list of propagation sub-graphs – one per rollout.
    """
    nodes = list(G)
    instances = []
    for _ in range(num_instances):
        seeds = random.sample(nodes, k=min(seed_count, len(nodes)))
        activated, log = simulate_ic(G, set(seeds))
        H = nx.DiGraph()
        H.add_nodes_from(activated)
        for e in log:
            H.add_edge(e["source"], e["target"], weight=e["prob_used"])
        instances.append(H)
    return instances
