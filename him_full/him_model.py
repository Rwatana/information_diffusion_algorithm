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

from hyperbolic_utils import (
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

        # rotation parameters (two sets: structure S and T; propagation S_d and T_d)
        theta_init = torch.zeros(dim // 2)
        self.theta_Ss = nn.Parameter(theta_init.clone())
        self.theta_Ts = nn.Parameter(theta_init.clone())
        self.theta_Sd = nn.Parameter(theta_init.clone())
        self.theta_Td = nn.Parameter(theta_init.clone())

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
        emb = self.embeddings  # (N, d+1)

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

        # negative sampling
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

        # unify edges across graphs
        all_edges = []
        for g in prop_graphs:
            all_edges.extend(g.edges)
        if not all_edges:
            return torch.tensor(0.0, device=device)

        sample_idx = random.sample(
            range(len(all_edges)), k=min(batch_edges, len(all_edges))
        )
        u_pos = torch.tensor([all_edges[i][0] for i in sample_idx], dtype=torch.long, device=device)
        v_pos = torch.tensor([all_edges[i][1] for i in sample_idx], dtype=torch.long, device=device)

        deg = torch.tensor(
                [G.out_degree(u) + 1e-3 for u in u_pos.cpu().tolist()],
                device=device,
             )
        w = 1.0 / deg

        # positive & negative
        pos_score = self._edge_score(
            u_pos, v_pos, self.theta_Sd, self.theta_Td, w
        )
        pos_loss = F.softplus(-pos_score).mean()

        u_neg = u_pos.repeat_interleave(self.neg_samples)
        num_neg = u_neg.shape[0]
        v_neg = torch.randint(0, self.num_nodes, (num_neg,), device=device)
        neg_score = self._edge_score(
            u_neg, v_neg, self.theta_Sd, self.theta_Td, w.repeat_interleave(self.neg_samples)
        )
        neg_loss = F.softplus(neg_score).mean()
        
        # propagation likelihood loss
        # P_G^D = P + I in Eq.10
        # where P is the positive log-likelihood and I is the influence regularization
        # loss = - ( P + I ) in Eq.10
        # P + I is the total likelihood loss
        # P + I = P_G^D
        # where P_G^D is the propagation likelihood loss
        # P_G^D = P + I

        prop_likelihood_loss = pos_loss + neg_loss

        # influence regularization – pull influencers toward origin
        emb = self.embeddings
        origin = torch.zeros(self.dim + 1, device=device)
        origin[0] = math.sqrt(self.gamma)
        d2_origin = lorentz_distance2(emb[u_pos], origin, gamma=self.gamma).squeeze(-1)

        # alpha = torch.sqrt(deg / deg.max())
        # reg = (alpha * torch.log(torch.sigmoid(d2_origin))).mean()
        
                
        # Calculate alpha_u = sqrt(d_u / d_max)
        # d_u is the degree of user u in the social network G.
        # d_max is the maximum degree in G.
        # For batch processing, d_max might be approximated within the batch or precomputed.
        # The current code uses G.out_degree for u_pos and then u_pos's max degree for d_max.
        # This is a practical approximation if G is very large.
        degrees_of_u_pos_in_G = torch.tensor(
            [G.out_degree(u_node.item()) + 1e-3 for u_node in u_pos], # Use .item() to get Python int for G.out_degree
            device=device,
        )
        # For d_max, ideally, it should be the max degree in the entire graph G.
        # If G is static, precompute: d_max_val = max(d for n, d in G.out_degree()) + 1e-3
        # For this example, let's assume d_max_val is available or approximated as in the code:
        d_max_approximation = degrees_of_u_pos_in_G.max() + 1e-9 # Add epsilon for stability if max is 0
        if d_max_approximation == 0: d_max_approximation = torch.tensor(1.0, device=device) # Avoid division by zero

        alpha = torch.sqrt(degrees_of_u_pos_in_G / d_max_approximation)

        # The regularization loss term to be minimized
        influence_regularization_loss = (alpha * d2_origin).mean()
        
        # # This is the term we want to add to the NEGATIVE log-likelihood (the loss)
        # # to achieve the "pull closer to origin" effect.
        # influence_pull_loss = (alpha * d2_origin).mean() # Minimize this
        # # Negative sign because loss = - ( P + I ) in Eq.10
        # return pos_loss + neg_loss + influence_pull_loss

        return prop_likelihood_loss + influence_regularization_loss



    # ------------- training ---------------- #
    # def fit(
    #     self,
    #     G: nx.DiGraph,
    #     propagations: List[nx.DiGraph],
    #     epochs: int = 200,
    #     lr: float = 5e-3,
    #     verbose: bool = True,
    # ):
    #     """Riemannian SGD training loop."""
    #     # Use standard Adam, treated as Euclidean on parameters; projection keeps on manifold.
    #     opt = torch.optim.Adam(self.parameters(), lr=lr)
    #     for epoch in range(1, epochs + 1):
    #         opt.zero_grad()
    #         loss_s = self._structure_loss(G)
    #         loss_p = self._propagation_loss(G, propagations)
    #         loss = loss_s + loss_p
    #         loss.backward()
    #         opt.step()

    #         # project euclidean part back to Lorentz manifold
    #         with torch.no_grad():
    #             self.eucl.data.copy_(self.eucl.data)  # placeholder; nothing extra needed
    #         if verbose and epoch % 20 == 0:
    #             print(f"Epoch {epoch}/{epochs}  loss={loss.item():.4f}")
    #     return self
    def fit(
        self,
        G: nx.DiGraph,
        propagations: List[nx.DiGraph],
        epochs: int = 200,
        lr: float = 5e-3,
        verbose: bool = True,
        ) -> "HIMModel":
        """Riemannian SGD training loop."""
        # 埋め込み履歴を収集するリストを用意
        self.embeddings_history: list[torch.Tensor] = []

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            loss_s = self._structure_loss(G)
            loss_p = self._propagation_loss(G, propagations)
            loss = loss_s + loss_p
            loss.backward()
            opt.step()

             # project euclidean part back to Lorentz manifold
            with torch.no_grad():
                 self.eucl.data.copy_(self.eucl.data)  # placeholder; nothing extra needed

            # ========== ここで埋め込みを保存 ==========
            with torch.no_grad():
                # self.embeddings は (N, d+1) の Lorentz 埋め込み
                self.embeddings_history.append(self.embeddings.detach().cpu().clone())
            # =======================================

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}  loss={loss.item():.4f}")
        return self

    

    # ------------- LDO & helpers ----------- #
    def compute_LDO(self):
        emb = self.embeddings
        origin = torch.zeros(self.dim + 1, device=device)
        origin[0] = math.sqrt(self.gamma)
        d2 = lorentz_distance2(emb, origin, gamma=self.gamma).squeeze(-1)
        return d2.detach().cpu()
