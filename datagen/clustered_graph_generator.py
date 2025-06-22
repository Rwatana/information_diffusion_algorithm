import networkx as nx

def generate_clustered_graph(num_nodes, p_intra=0.2, p_inter=0.01, seed=None):
    """
    Generates a graph with two distinct clusters using the stochastic block model.

    Args:
        num_nodes (int): The total number of nodes in the graph.
        p_intra (float): The probability of an edge within a cluster.
        p_inter (float): The probability of an edge between the two clusters.
        seed (int, optional): Seed for the random number generator for reproducibility.

    Returns:
        nx.Graph: The generated graph with two clusters.
    """
    if num_nodes < 2:
        raise ValueError("Number of nodes must be at least 2.")

    # ノードを2つのクラスターに分割
    cluster1_nodes = num_nodes // 2
    cluster2_nodes = num_nodes - cluster1_nodes
    sizes = [cluster1_nodes, cluster2_nodes]

    # クラスター内・クラスター間の接続確率を行列で定義
    probs = [
        [p_intra, p_inter],
        [p_inter, p_intra]
    ]

    # Stochastic Block Modelを用いてグラフを生成
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return G