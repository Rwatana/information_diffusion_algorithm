"""
run_experiment.py

End‑to‑end pipeline for HIM: data generation, embedding learning,
adaptive seed selection, printing final seed set.
"""
import os
import sys
import argparse

# data_utils.py がある親ディレクトリ (この場合は 'datagen' フォルダのある階層) をPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # pagerank_experiment から見て一つ上の階層
from datagen.data_utils import generate_graph, generate_propagations
from him_model import HIMModel
from seed_selection import adaptive_sliding_window
import torch

def main():
    p = argparse.ArgumentParser(description="Full HIM demo")
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--edge_density", type=float, default=0.01)
    p.add_argument("--seed_count", type=int, default=10)
    p.add_argument("--prop_instances", type=int, default=30)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--ic_prob", type=float, default=0.05)
    args = p.parse_args()

    print("[1/4] Generating base graph …")
    G = generate_graph(args.nodes, args.edge_density)

    print("[2/4] Simulating propagation instances …")
    propagations = generate_propagations(G,
                                         seed_count=args.seed_count,
                                         num_instances=args.prop_instances,
                                         ic_prob=args.ic_prob)

    print("[3/4] Training HIM embeddings … (may take a while)")
    model = HIMModel(num_nodes=args.nodes, dim=args.dim).fit(G, propagations,
                                                             epochs=args.epochs,
                                                             verbose=True)
    torch.save(model.embeddings_history, "traj.pt")

    print("[4/4] Running Adaptive Sliding Window …")
    emb = model.embeddings.detach()
    seeds = adaptive_sliding_window(G, emb, k=args.seed_count, beta=args.beta)
    print(f"Selected seeds (k={args.seed_count}): {seeds}")

if __name__ == "__main__":
    main()
