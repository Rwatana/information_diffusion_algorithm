"""
seed_selection.py

Adaptive Sliding Window (ASW) implementation as Algorithm 1
in the HIM paper.
"""

import heapq
import math
from typing import List

import networkx as nx
import torch

from hyperbolic_utils import lorentz_distance2, device

def adaptive_sliding_window(
    G: nx.DiGraph,
    embeddings: torch.Tensor,
    k: int,
    beta: float = 3.0,
    gamma: float = 1.0,
) -> List[int]:
    """Return k seed node indices using ASW."""
    origin = torch.zeros(embeddings.size(1), device=device)
    origin[0] = math.sqrt(gamma)

    # LDO scores (lower is better)
    d2_origin = lorentz_distance2(embeddings.to(device), origin, gamma=gamma).squeeze(-1)
    scores = d2_origin.cpu().tolist()

    nodes = list(range(G.number_of_nodes()))
    nodes_sorted = [n for n, _ in sorted(zip(nodes, scores), key=lambda kv: kv[1])]

    w_size = int(beta * k)
    window = []
    seeds = []
    next_idx = 1  # first seed uses nodes_sorted[0]
    current = nodes_sorted[0]

    # initialize window
    while len(window) < w_size and next_idx < len(nodes_sorted):
        u = nodes_sorted[next_idx]
        heapq.heappush(window, (scores[u], u))
        next_idx += 1

    while len(seeds) < k:
        seeds.append(current)
        neigh = set(G.successors(current)) | set(G.predecessors(current))
        cand_keys = {u for _, u in window}
        C = neigh & cand_keys
        if C:
            # update scores for nodes in C according to Eq.(11)(12)
            deg_c = G.out_degree(current) + G.in_degree(current) + 1
            # pre‑compute denominator
            emb_c = embeddings[current].unsqueeze(0).to(device)
            emb_C = embeddings[list(C)].to(device)
            d2_cv = lorentz_distance2(emb_c.expand_as(emb_C), emb_C, gamma=gamma).squeeze(-1)
            weights = torch.exp(1.0 / d2_cv)
            denom = weights.sum().item() + 1e-9

            for w_cv, v, d2 in zip(weights, C, d2_cv):
                w_cv = w_cv.item() / denom
                scores[v] += (w_cv / deg_c) * scores[current]
                # update heap by pushing new key; duplicates fine, we will pop later
                heapq.heappush(window, (scores[v], v))

        # pick next minimum (skip already selected)
        while window:
            sc, u = heapq.heappop(window)
            if u not in seeds:
                current = u
                break
        else:
            # window exhausted
            if next_idx < len(nodes_sorted):
                current = nodes_sorted[next_idx]
                next_idx += 1
            else:
                break

        # refill window
        while len(window) < w_size and next_idx < len(nodes_sorted):
            u = nodes_sorted[next_idx]
            if u not in seeds:
                heapq.heappush(window, (scores[u], u))
            next_idx += 1

    return seeds[:k]
