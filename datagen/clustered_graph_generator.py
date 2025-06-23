import networkx as nx

def generate_clustered_graph(num_nodes, num_clusters, p_intra=0.2, p_inter=0.01, seed=None):
    """
    指定されたクラスター数でグラフを生成します。

    Args:
        num_nodes (int): グラフの総ノード数
        num_clusters (int): 作成するクラスターの数
        p_intra (float): クラスター内の辺の接続確率
        p_inter (float): クラスター間の辺の接続確率
        seed (int, optional): 乱数生成器のシード値

    Returns:
        nx.Graph: 生成されたグラフ
    """
    if num_nodes < num_clusters:
        raise ValueError("ノード数はクラスター数以上である必要があります。")

    if num_clusters <= 0:
        raise ValueError("クラスター数は1以上である必要があります。")

    # 各クラスターのノード数を計算
    sizes = [num_nodes // num_clusters for _ in range(num_clusters)]
    remainder = num_nodes % num_clusters
    for i in range(remainder):
        sizes[i] += 1
    
    # 接続確率を行列で定義
    probs = [[p_intra if i == j else p_inter for j in range(num_clusters)] for i in range(num_clusters)]

    # Stochastic Block Modelを用いてグラフを生成
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    return G