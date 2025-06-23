import streamlit as st
import networkx as nx
import json
import os
import sys
import yaml
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import pandas as pd

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from him_full.him_model import HIMModel
from him_full.hyperbolic_utils import lorentz_distance2

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
st.title("双曲埋め込みの内的な性質評価")
st.write("グラフを選択し、双曲埋め込みを計算後、その埋め込みが元のグラフの性質をどれだけ保持しているかを複数の指標で評価します。")

# --- Helper Functions for Evaluation ---
def poincare_dist(u, v):
    """ポアンカレ球モデルにおける距離を計算 (バッチ処理対応)"""
    sq_u_norm = torch.sum(u * u, dim=-1)
    sq_v_norm = torch.sum(v * v, dim=-1)
    sq_dist = torch.sum(torch.pow(u - v, 2), dim=-1)
    # 浮動小数点数誤差によるクリップ
    cosh_arg = 1 + 2 * sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))
    cosh_arg = torch.clamp(cosh_arg, min=1.0)
    return torch.acosh(cosh_arg)

def calculate_distortion(G, embeddings, num_samples=2000):
    """1. 歪み（Distortion）を計算"""
    nodes = list(G.nodes())
    if len(nodes) == 0: return 0, 0
    
    # ノードペアをサンプリング
    pairs = [random.sample(nodes, 2) for _ in range(num_samples)]
    u_indices, v_indices = zip(*pairs)
    
    # グラフ距離（最短経路長）を計算
    graph_distances = [nx.shortest_path_length(G, source=u, target=v) for u, v in pairs]
    
    # 双曲距離を計算
    u_emb = embeddings[list(u_indices)]
    v_emb = embeddings[list(v_indices)]
    hyperbolic_distances = poincare_dist(u_emb, v_emb).numpy()
    
    # 相関係数を計算
    spearman_corr, _ = spearmanr(graph_distances, hyperbolic_distances)
    return spearman_corr

def calculate_hierarchy_reconstruction(G, embeddings):
    """2. 階層構造の再現性を評価"""
    if len(G.nodes) == 0: return 0
    
    # グラフの中心性（次数）を計算
    centralities = np.array(list(nx.degree_centrality(G).values()))
    
    # 双曲空間での原点からの距離（ノルム）を計算
    # Poincare Diskではユークリッドノルムが原点からの距離と相関
    norms = torch.norm(embeddings, p=2, dim=1).numpy()
    
    # スピアマンの順位相関係数を計算 (中心性が高いほどノルムが小さいはずなので、負の相関を期待)
    spearman_corr, _ = spearmanr(centralities, norms)
    return spearman_corr

def calculate_reconstruction_error(G, model):
    """3. 再構成誤差を計算"""
    nodes = list(G.nodes())
    num_nodes = len(nodes)
    if num_nodes == 0: return 0

    # ポジティブサンプル（存在するエッジ）
    pos_edges = list(G.edges())
    
    # ネガティブサンプル（存在しないエッジ）
    neg_edges = []
    while len(neg_edges) < len(pos_edges):
        u, v = random.sample(nodes, 2)
        if not G.has_edge(u, v):
            neg_edges.append((u, v))
            
    edges = pos_edges + neg_edges
    y_true = [1] * len(pos_edges) + [0] * len(neg_edges)
    
    u_indices, v_indices = zip(*edges)
    u_idx = torch.tensor(list(u_indices), dtype=torch.long)
    v_idx = torch.tensor(list(v_indices), dtype=torch.long)
    
    # モデルを使ってエッジのスコアを計算
    with torch.no_grad():
        # _edge_scoreは重みを必要とするが、評価では無視して距離とバイアスのみで計算
        dummy_w = torch.ones(len(u_idx))
        scores = model._edge_score(u_idx, v_idx, model.theta_Ss, model.theta_Ts, dummy_w).numpy()
    
    return average_precision_score(y_true, scores)

def plot_poincare(embeddings, G):
    """4. ポアンカレ円盤にプロット"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 円盤を描画
    circle = patches.Circle((0, 0), radius=1, fill=False, color='black', linewidth=1.5)
    ax.add_patch(circle)
    
    # 埋め込みベクトル（最初の2次元）をプロット
    emb_2d = embeddings[:, :2].numpy()
    
    # クラスター情報で色分け
    node_colors = []
    if G.nodes and 'block' in next(iter(G.nodes(data=True)))[1]:
        clusters = [G.nodes[i].get('block', 0) for i in G.nodes()]
        num_clusters = len(set(clusters))
        cmap = plt.get_cmap("viridis", num_clusters)
        node_colors = [cmap(c) for c in clusters]
    else:
        node_colors = ['#1f77b4'] * len(emb_2d)

    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=node_colors, s=50, alpha=0.8)
    
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    return fig

# --- General Helper Functions ---
def get_saved_graphs():
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_graphs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        return []
    return sorted([d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))], reverse=True)

def load_graph(graph_name):
    graph_dir = os.path.join(os.path.dirname(__file__), 'saved_graphs', graph_name)
    graph_file = os.path.join(graph_dir, 'graph_data.json')
    if os.path.exists(graph_file):
        with open(graph_file, 'r') as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        if not G.is_directed():
            G = G.to_directed()
        return G
    return None

# --- SIDEBAR ---
st.sidebar.title("設定")
st.sidebar.header("ステップ1: 評価するグラフを選択")
saved_graphs = get_saved_graphs()
if not saved_graphs:
    st.sidebar.warning("保存されているグラフがありません。")
    st.stop()
selected_graph_name = st.sidebar.selectbox("グラフを選択", options=saved_graphs)

st.sidebar.header("ステップ2: 双曲埋め込みのパラメータ設定")
try:
    default_params_path = os.path.join(os.path.dirname(__file__), '../../him_full/params.yaml')
    with open(default_params_path, 'r') as f:
        default_params = yaml.safe_load(f)
except Exception:
    default_params = {'dim': 10, 'epochs': 100, 'lr': 0.1, 'neg_samples': 10}

dim = st.sidebar.slider("埋め込み次元数 (dim)", 2, 64, default_params.get('dim', 10), step=2)
epochs = st.sidebar.slider("学習エポック数 (epochs)", 10, 500, default_params.get('epochs', 100))
lr = st.sidebar.number_input("学習率 (lr)", 0.001, 1.0, default_params.get('lr', 0.1), format="%.3f")
neg_samples = st.sidebar.slider("ネガティブサンプル数", 1, 50, default_params.get('neg_samples', 10))

if st.sidebar.button("埋め込み計算と評価を実行"):
    G = load_graph(selected_graph_name)
    if G:
        st.session_state['eval_graph'] = G
        st.session_state['eval_graph_name'] = selected_graph_name

        with st.spinner(f"{epochs}エポックで双曲埋め込みを学習中..."):
            model = HIMModel(num_nodes=G.number_of_nodes(), dim=dim, neg_samples=neg_samples)
            model.fit(G=G, propagations=[], epochs=epochs, lr=lr, verbose=False)
            st.session_state['embeddings'] = model.embeddings.detach().cpu()
            st.session_state['model'] = model # モデル全体を保存
        
        st.sidebar.success("埋め込み計算が完了しました。")
        st.session_state['evaluation_ready'] = True
    else:
        st.sidebar.error("グラフの読み込みに失敗しました。")

# --- MAIN PANEL ---
if st.session_state.get('evaluation_ready'):
    G = st.session_state['eval_graph']
    graph_name = st.session_state['eval_graph_name']
    embeddings = st.session_state['embeddings']
    model = st.session_state['model']
    
    st.header(f"評価結果: `{graph_name}`")

    with st.expander("1. 歪み（Distortion）の評価", expanded=True):
        with st.spinner("歪みを計算中..."):
            spearman_corr = calculate_distortion(G.to_undirected(), embeddings)
            st.metric("スピアマンの順位相関係数", f"{spearman_corr:.4f}")
            st.info("値が1に近いほど、元のグラフの距離構造が双曲空間で忠実に再現されています。")

    with st.expander("2. 階層構造の再現性評価", expanded=True):
        with st.spinner("階層性の再現度を計算中..."):
            spearman_corr_h = calculate_hierarchy_reconstruction(G, embeddings)
            st.metric("次数中心性と双曲ノルムの相関係数", f"{spearman_corr_h:.4f}")
            st.info("値が-1に近いほど、「次数の高い中心的なノードが原点近くに配置される」という理想的な階層構造が再現されています。")

    with st.expander("3. 再構成誤差（Reconstruction Error）", expanded=True):
        with st.spinner("再構成誤差を計算中..."):
            ap_score = calculate_reconstruction_error(G, model)
            st.metric("平均適合率 (Average Precision)", f"{ap_score:.4f}")
            st.info("値が1に近いほど、埋め込みベクトルから元のエッジの有無を正確に予測できます。")

    with st.expander("4. 可視化による定性的評価", expanded=True):
        with st.spinner("可視化を生成中..."):
            fig = plot_poincare(embeddings, G)
            st.pyplot(fig)
else:
    st.info("サイドバーでグラフを選択し、パラメータを設定してから「埋め込み計算と評価を実行」ボタンを押してください。")