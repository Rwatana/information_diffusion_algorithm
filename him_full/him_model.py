"""
him_model.py

Full implementation of the Hyperbolic Influence Maximization
embedding learner (Hyperbolic Influence Representation module).
"""

import math
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .hyperbolic_utils import (
    project_to_lorentz,
    lorentz_distance2,
    rotate,
    device,
)

class HIMModel(nn.Module):
    """Hyperbolic embedding learner following HIM paper."""

    def __init__(
        self,
        num_nodes: int,
        dim: int = 64,
        gamma: float = 1.0,
        neg_samples: int = 5,
    ):
        super().__init__()
        assert (
            dim % 2 == 0
        ), "Embedding dimension must be even for block‑diag rotations"
        self.num_nodes = num_nodes
        self.dim = dim
        self.gamma = gamma
        self.neg_samples = neg_samples

        # learnable Euclidean parameters (n, dim)
        eucl = torch.randn(num_nodes, dim) * 0.01
        self.eucl = nn.Parameter(eucl)

        # learnable biases
        self.bias = nn.Parameter(torch.zeros(num_nodes))

        # rotation parameters
        theta_init = torch.zeros(dim // 2)
        self.theta_Ss = nn.Parameter(theta_init.clone())
        self.theta_Ts = nn.Parameter(theta_init.clone())
        self.theta_Sd = nn.Parameter(theta_init.clone())
        self.theta_Td = nn.Parameter(theta_init.clone())

        # 履歴を保存するリストをここで初期化
        self.embeddings_history: List[torch.Tensor] = []
        self.ldo_history: List[torch.Tensor] = []


    # ------------- helper properties ------------- #
    @property
    def embeddings(self) -> torch.Tensor:
        """Return Lorentz model embeddings (n+1 dims)."""
        return project_to_lorentz(self.eucl, gamma=self.gamma)

    def _edge_score(
        self,
        u_idx: torch.Tensor,
        v_idx: torch.Tensor,
        theta_src: torch.Tensor,
        theta_dst: torch.Tensor,
        w_uv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute V_uv value in equations (4)/(7)."""
        emb = self.embeddings

        x_u = emb[u_idx]
        x_v = emb[v_idx]

        xSu = rotate(x_u, theta_src)
        xTv = rotate(x_v, theta_dst)

        d2 = lorentz_distance2(xSu, xTv, gamma=self.gamma).squeeze(-1)
        b_u = self.bias[u_idx]
        b_v = self.bias[v_idx]

        return -w_uv * d2 + b_u + b_v

    # ------------- losses ---------------- #
    def _structure_loss(self, G: nx.DiGraph, batch_size: int = 1024) -> torch.Tensor:
        """Compute network structure loss P from Eq.(4)/(5)."""
        edges = list(G.edges)
        if not edges:
            return torch.tensor(0.0, device=device)

        sample_idx = random.sample(range(len(edges)), k=min(batch_size, len(edges)))
        u_pos = torch.tensor([edges[i][0] for i in sample_idx], dtype=torch.long, device=device)
        v_pos = torch.tensor([edges[i][1] for i in sample_idx], dtype=torch.long, device=device)

        deg = torch.tensor([G.out_degree(u) + 1e-3 for u in u_pos.cpu().tolist()], device=device)
        w = 1.0 / deg

        pos_score = self._edge_score(u_pos, v_pos, self.theta_Ss, self.theta_Ts, w)
        pos_loss = F.softplus(-pos_score).mean()

        num_neg = self.neg_samples * len(u_pos)
        u_neg = u_pos.repeat_interleave(self.neg_samples)
        v_neg = torch.randint(0, self.num_nodes, (num_neg,), device=device)

        neg_score = self._edge_score(u_neg, v_neg, self.theta_Ss, self.theta_Ts, w.repeat_interleave(self.neg_samples))
        neg_loss = F.softplus(neg_score).mean()

        return pos_loss + neg_loss

    def _propagation_loss(
        self, G: nx.DiGraph, prop_graphs: List[nx.DiGraph], batch_edges: int = 2048
      ) -> torch.Tensor:
        """Compute loss P_G^D and I_G^D for batch of propagation edges."""
        if not prop_graphs:
            return torch.tensor(0.0, device=device)

        all_edges = []
        for g in prop_graphs:
            all_edges.extend(g.edges)
        if not all_edges:
            return torch.tensor(0.0, device=device)

        sample_idx = random.sample(range(len(all_edges)), k=min(batch_edges, len(all_edges)))
        u_pos = torch.tensor([all_edges[i][0] for i in sample_idx], dtype=torch.long, device=device)
        v_pos = torch.tensor([all_edges[i][1] for i in sample_idx], dtype=torch.long, device=device)

        deg = torch.tensor([G.out_degree(u) + 1e-3 for u in u_pos.cpu().tolist()], device=device)
        w = 1.0 / deg

        pos_score = self._edge_score(u_pos, v_pos, self.theta_Sd, self.theta_Td, w)
        pos_loss = F.softplus(-pos_score).mean()

        u_neg = u_pos.repeat_interleave(self.neg_samples)
        num_neg = u_neg.shape[0]
        v_neg = torch.randint(0, self.num_nodes, (num_neg,), device=device)
        neg_score = self._edge_score(u_neg, v_neg, self.theta_Sd, self.theta_Td, w.repeat_interleave(self.neg_samples))
        neg_loss = F.softplus(neg_score).mean()
        
        prop_likelihood_loss = pos_loss + neg_loss

        emb = self.embeddings
        origin = torch.zeros(self.dim + 1, device=device)
        origin[0] = math.sqrt(self.gamma)
        d2_origin = lorentz_distance2(emb[u_pos], origin, gamma=self.gamma).squeeze(-1)

        degrees_of_u_pos_in_G = torch.tensor(
            [G.out_degree(u_node.item()) + 1e-3 for u_node in u_pos],
            device=device,
        )
        d_max_approximation = degrees_of_u_pos_in_G.max() + 1e-9
        if d_max_approximation == 0: d_max_approximation = torch.tensor(1.0, device=device)

        alpha = torch.sqrt(degrees_of_u_pos_in_G / d_max_approximation)
        influence_regularization_loss = (alpha * d2_origin).mean()

        return prop_likelihood_loss + influence_regularization_loss

    # ------------- training ---------------- #
    def fit(
        self,
        G: nx.DiGraph,
        propagations: List[nx.DiGraph],
        epochs: int = 200,
        lr: float = 5e-3,
        verbose: bool = True,
        ) -> "HIMModel":
        """Riemannian SGD training loop."""
        # 履歴リストをクリア
        self.embeddings_history.clear()
        self.ldo_history.clear()

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            loss_s = self._structure_loss(G)
            loss_p = self._propagation_loss(G, propagations)
            loss = loss_s + loss_p
            loss.backward()
            opt.step()

            with torch.no_grad():
                self.eucl.data.copy_(self.eucl.data)

            # エポックごとに埋め込みとLDOを保存
            with torch.no_grad():
                self.embeddings_history.append(self.embeddings.detach().cpu().clone())
                self.ldo_history.append(self.compute_LDO().detach().cpu().clone())

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}  loss={loss.item():.4f}")
        return self

    # ------------- LDO & helpers ----------- #
    def compute_LDO(self) -> torch.Tensor:
        """Compute Lorentz Distance from Origin (LDO)."""
        emb = self.embeddings
        origin = torch.zeros(self.dim + 1, device=device)
        origin[0] = math.sqrt(self.gamma)
        d2 = lorentz_distance2(emb, origin, gamma=self.gamma).squeeze(-1)
        return d2.detach().cpu()
