"""
degree_centrality.py

Implements Degree Centrality based seed selection.
"""
import networkx as nx

def get_top_k_nodes_from_scores(scores, k):
    """
    スコアに基づいて上位k個のノードを返す (共通ヘルパー関数)
    """
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_k = [node_id for node_id, score in sorted_nodes[:k]]
    return top_k

def select_seeds_by_degree_centrality(graph: nx.DiGraph, k: int, use_out_degree=True):
    """
    次数中心性アルゴリズムを実行し、上位k個のノードを返す
    use_out_degree: Trueなら出次数中心性、Falseなら入次数中心性を使用
    """
    print("Calculating Degree Centrality scores...")
    if use_out_degree:
        print("(Using Out-Degree Centrality)")
        degree_scores = nx.out_degree_centrality(graph)
    else:
        print("(Using In-Degree Centrality)")
        degree_scores = nx.in_degree_centrality(graph)

    seeds = get_top_k_nodes_from_scores(degree_scores, k)
    print(f"Top {k} seeds selected by Degree Centrality: {seeds}")
    return seeds