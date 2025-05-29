"""
run_pagerank_experiment.py

Main script to generate a graph and run PageRank algorithm.
Located in the pagerank_experiment folder.
"""
import argparse
import sys
import os

# data_utils.py がある親ディレクトリ (この場合は 'datagen' フォルダのある階層) をPythonのパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # pagerank_experiment から見て一つ上の階層
from datagen.data_utils import generate_graph, generate_propagations
from pagerank import select_seeds_by_pagerank

def main():
    p = argparse.ArgumentParser(description="PageRank Algorithm Runner")
    p.add_argument("--nodes", type=int, default=1000, help="Number of nodes in the graph.")
    p.add_argument("--edge_density", type=float, default=0.01, help="Edge density of the graph.")
    p.add_argument("--seed_count", type=int, default=10, help="Number of seed nodes to select (k).")

    args = p.parse_args()

    print("Step 1: Generating base graph for PageRank experiment...")
    G = generate_graph(args.nodes, args.edge_density, directed=True)
    print(f"Graph generated with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print(f"\nStep 2: Running PageRank algorithm to select {args.seed_count} seeds...")
    selected_seeds = select_seeds_by_pagerank(G, args.seed_count)

    print("\nPageRank experiment finished.")
    # print(f"Selected seeds: {selected_seeds}")

if __name__ == "__main__":
    main()