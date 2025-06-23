import streamlit as st
import networkx as nx
from pyvis.network import Network
import json
import os
from datetime import datetime
import sys
import colorsys
import random

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datagen.clustered_graph_generator import generate_clustered_graph

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
st.title("クラスターグラフの生成・読み込みと可視化")

# --- HELPER FUNCTIONS ---

def generate_colors(n):
    """色を生成する関数"""
    colors = []
    for i in range(n):
        hue = i / n
        lightness = 0.5
        saturation = 0.8
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f'#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors

def display_graph(G, params):
    """グラフ情報と可視化をメインパネルに表示する関数"""
    st.header("グラフ情報")
    col1, col2, col3 = st.columns(3)
    col1.metric("ノード数", G.number_of_nodes())
    col2.metric("エッジ数", G.number_of_edges())
    col3.metric("有向グラフか？", "はい" if G.is_directed() else "いいえ")

    st.subheader("伝播確率の設定値")
    if "prob_ranges" in params:
        ranges = params["prob_ranges"]
        col1, col2 = st.columns(2)
        col1.info(f"**クラスター内:** `{ranges['intra_min']:.3f}` ~ `{ranges['intra_max']:.3f}`")
        col2.info(f"**クラスター外:** `{ranges['inter_min']:.3f}` ~ `{ranges['inter_max']:.3f}`")

    st.header("グラフの可視化")
    net = Network(notebook=True, cdn_resources='in_line', height="750px", width="100%", directed=True)
    
    for node, attrs in G.nodes(data=True):
        cluster_id = attrs.get('block', 0)
        net.add_node(node, label=str(node), group=cluster_id)

    # --- ここからが変更点 ---
    # 属性名を'weight'に変更
    for source, target, attrs in G.edges(data=True):
        weight = attrs.get('weight') 
        title = f"Prob: {weight:.4f}" if weight is not None else ""
        net.add_edge(source, target, title=title, value=weight)
    # --- ここまでが変更点 ---

    try:
        if G.nodes and 'block' in next(iter(G.nodes(data=True)))[1]:
            num_clusters = len(set(nx.get_node_attributes(G, 'block').values()))
            colors = generate_colors(num_clusters)
            for node in net.nodes:
                cluster_id = G.nodes[node['id']].get('block',0)
                node['color'] = colors[cluster_id % len(colors)]
    except (StopIteration, IndexError, KeyError):
        pass

    net.show_buttons(filter_=['physics'])

    html_file = "clustered_graph_visualization.html"
    net.save_graph(html_file)
    st.components.v1.html(open(html_file, 'r', encoding='utf-8').read(), height=800, scrolling=True)

def get_saved_graphs():
    """saved_graphsディレクトリからグラフのリストを取得する"""
    save_dir = os.path.join(os.path.dirname(__file__), 'saved_graphs')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        return []
    return [d for d in os.listdir(save_dir) if os.path.isdir(os.path.join(save_dir, d))]

# --- SIDEBAR ---

with st.sidebar.expander("グラフ設定", expanded=True):
    st.header("トポロジー設定")
    num_nodes_input = st.slider("ノード数", 5, 500, 50, key="num_nodes")
    num_clusters_input = st.slider("クラスター数", 2, 10, 3, key="num_clusters")
    p_intra_input = st.slider("クラスター内 結合確率", 0.0, 1.0, 0.2, key="p_intra")
    p_inter_input = st.slider("クラスター間 結合確率", 0.0, 1.0, 0.01, key="p_inter")
    
    st.header("伝播確率の設定")
    min_intra_prob = st.slider("クラスター内 最小伝播確率", 0.0, 1.0, 0.01, key="min_intra")
    max_intra_prob = st.slider("クラスター内 最大伝播確率", 0.0, 1.0, 0.2, key="max_intra")
    min_inter_prob = st.slider("クラスター外 最小伝播確率", 0.0, 1.0, 0.0, key="min_inter")
    max_inter_prob = st.slider("クラスター外 最大伝播確率", 0.0, 1.0, 0.01, key="max_inter")

    if st.button("新しいグラフを生成 (確率付き)", key="generate_graph"):
        G_undirected = generate_clustered_graph(
            num_nodes=num_nodes_input,
            num_clusters=num_clusters_input,
            p_intra=p_intra_input,
            p_inter=p_inter_input,
            seed=0
        )
        
        G_directed = G_undirected.to_directed()

        # --- ここからが変更点 ---
        # 各エッジに'weight'という名前で伝播確率を割り当て
        for u, v in G_undirected.edges():
            if G_undirected.nodes[u]['block'] == G_undirected.nodes[v]['block']:
                min_prob, max_prob = min_intra_prob, max_intra_prob
            else:
                min_prob, max_prob = min_inter_prob, max_inter_prob
            
            prob_uv = random.uniform(min_prob, max_prob)
            prob_vu = random.uniform(min_prob, max_prob)
            
            G_directed.edges[u, v]['weight'] = prob_uv
            G_directed.edges[v, u]['weight'] = prob_vu
        # --- ここまでが変更点 ---
            
        st.session_state['graph_object'] = G_directed
        st.session_state['params'] = {
            "prob_ranges": {
                "intra_min": min_intra_prob, "intra_max": max_intra_prob,
                "inter_min": min_inter_prob, "inter_max": max_inter_prob,
            }
        }
        st.session_state['graph_loaded_or_generated'] = True

# --- グラフ読み込みセクション ---
with st.sidebar.expander("グラフを読み込む"):
    saved_graphs = get_saved_graphs()
    if not saved_graphs:
        st.info("保存されているグラフがありません。")
    else:
        selected_graph = st.selectbox("読み込むグラフを選択してください", options=saved_graphs)
        if st.button("選択したグラフを読み込む"):
            graph_dir = os.path.join(os.path.dirname(__file__), 'saved_graphs', selected_graph)
            graph_file = os.path.join(graph_dir, 'graph_data.json')

            if os.path.exists(graph_file):
                with open(graph_file, 'r') as f:
                    graph_data = json.load(f)
                
                G = nx.node_link_graph(graph_data)
                
                if not G.is_directed():
                    G = G.to_directed()

                st.session_state['graph_object'] = G
                st.session_state['params'] = graph_data.get('params', {})
                st.session_state['graph_loaded_or_generated'] = True
                st.success(f"グラフ「{selected_graph}」を読み込みました。")
            else:
                st.error(f"{graph_file} が見つかりません。")

# --- グラフ保存セクション ---
if st.session_state.get('graph_loaded_or_generated'):
    with st.sidebar.expander("現在のグラフを保存"):
        default_name = f"clustered_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        graph_name = st.text_input("保存名を入力", default_name)

        if st.button("グラフを保存"):
            if graph_name and 'graph_object' in st.session_state:
                G_to_save = st.session_state['graph_object']
                if not G_to_save.is_directed():
                    G_to_save = G_to_save.to_directed()

                params_to_save = st.session_state.get('params', {})
                save_dir = os.path.join(os.path.dirname(__file__), 'saved_graphs', graph_name)
                os.makedirs(save_dir, exist_ok=True)

                data_to_save = nx.node_link_data(G_to_save)
                data_to_save['params'] = params_to_save
                
                def convert_to_json_serializable(obj):
                    if isinstance(obj, (set, frozenset)):
                        return [convert_to_json_serializable(item) for item in obj]
                    if isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [convert_to_json_serializable(elem) for elem in obj]
                    return obj

                serializable_data = convert_to_json_serializable(data_to_save)

                file_path = os.path.join(save_dir, "graph_data.json")
                with open(file_path, 'w') as f:
                    json.dump(serializable_data, f, indent=4)

                st.success(f"グラフを {save_dir} 内の graph_data.json に保存しました。")
            else:
                st.error("グラフ名が入力されていないか、保存対象のグラフが存在しません。")

# --- MAIN PANEL DISPLAY ---
if st.session_state.get('graph_loaded_or_generated'):
    display_graph(st.session_state['graph_object'], st.session_state.get('params', {}))
else:
    st.info("サイドバーでグラフを新規作成、または保存済みのグラフを読み込んでください。")