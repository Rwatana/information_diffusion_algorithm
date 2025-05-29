"""
run_degree_experiment.py

Main script to generate a graph and run Degree Centrality algorithm.
Located in the degree_centrality_experiment folder.
"""
import argparse
import sys
import os


# data_utils.py がある親ディレクトリ (この場合は 'datagen' フォルダのある階層) をPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # pagerank_experiment から見て一つ上の階層
from datagen.data_utils import generate_graph # もし datagen フォルダ内にあるなら
from degree_centrality import select_seeds_by_degree_centrality

def main():
    p = argparse.ArgumentParser(description="Degree Centrality Algorithm Runner")
    p.add_argument("--nodes", type=int, default=1000, help="Number of nodes in the graph.")
    p.add_argument("--edge_density", type=float, default=0.01, help="Edge density of the graph.")
    p.add_argument("--seed_count", type=int, default=10, help="Number of seed nodes to select (k).")
    p.add_argument("--degree_type", type=str, choices=['in', 'out'], default='out',
                       help="For degree centrality, use 'in' or 'out' degree (default: 'out').")

    args = p.parse_args()

    print("Step 1: Generating base graph for Degree Centrality experiment...")
    G = generate_graph(args.nodes, args.edge_density, directed=True)
    print(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print(f"\nStep 2: Running Degree Centrality ({args.degree_type}-degree) to select {args.seed_count} seeds...")
    use_out = True if args.degree_type == 'out' else False
    selected_seeds = select_seeds_by_degree_centrality(G, args.seed_count, use_out_degree=use_out)

    print("\nDegree Centrality experiment finished.")
    # print(f"Selected seeds: {selected_seeds}")

if __name__ == "__main__":
    main()