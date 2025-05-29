"""
pagerank.py

Implements PageRank based seed selection.
"""
import networkx as nx

def get_top_k_nodes_from_scores(scores, k):
    """
    スコアに基づいて上位k個のノードを返す (共通ヘルパー関数)
    """
    sorted_nodes = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_k = [node_id for node_id, score in sorted_nodes[:k]]
    return top_k

def select_seeds_by_pagerank(graph: nx.DiGraph, k: int):
    """
    PageRankアルゴリズムを実行し、上位k個のノードを返す
    """
    print("Calculating PageRank scores...")
    pagerank_scores = nx.pagerank(graph)
    seeds = get_top_k_nodes_from_scores(pagerank_scores, k)
    print(f"Top {k} seeds selected by PageRank: {seeds}")
    return seeds