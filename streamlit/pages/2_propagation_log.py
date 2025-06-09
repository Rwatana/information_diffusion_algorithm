import streamlit as st
import pandas as pd
import random
import sys
import os
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import json

# --- パス設定とモジュールのインポート ---
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from datagen.data_utils import simulate_ic
except (ImportError, ModuleNotFoundError):
    st.error("エラー: `datagen.data_utils` モジュールが見つかりません。ディレクトリ構造を確認してください。")
    st.stop()

# --- 定数とディレクトリ設定 ---
SAVE_DIR_NAME = "saved_graphs"
SAVE_DIR_PATH = os.path.join(current_file_dir, SAVE_DIR_NAME)
if not os.path.exists(SAVE_DIR_PATH):
    st.error(f"保存ディレクトリが見つかりません: {SAVE_DIR_PATH}")
    st.info("`1_graph_visualization.py` などのページでグラフを保存すると、自動的に作成されます。")
    st.stop()

# --- ヘルパー関数 ---
def load_graph_from_json(folder_name):
    filepath = os.path.join(SAVE_DIR_PATH, folder_name, 'graph_data.json')
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    except Exception as e:
        st.error(f"グラフ読込エラー: {e}")
        return None

def get_saved_graph_files():
    if not os.path.exists(SAVE_DIR_PATH): return []
    return sorted([d for d in os.listdir(SAVE_DIR_PATH) if os.path.isdir(os.path.join(SAVE_DIR_PATH, d))], reverse=True)

# --- セッションステートの初期化 ---
# このページ専用のキーを使用
if 'prop_log_graph' not in st.session_state:
    st.session_state.prop_log_graph = None
if 'prop_log_graph_name' not in st.session_state:
    st.session_state.prop_log_graph_name = "未選択"
# 比較表示用の結果保持
if 'prop_log_latest_results' not in st.session_state:
    st.session_state.prop_log_latest_results = None
if 'prop_log_previous_results' not in st.session_state:
    st.session_state.prop_log_previous_results = None

# --- サイドバー ---
st.sidebar.title("Propagation Analysis")
st.sidebar.header("Step 1: グラフを選択")
saved_files = get_saved_graph_files()

if not saved_files:
    st.sidebar.error("読み込み可能なグラフがありません。")
else:
    selected_file = st.sidebar.selectbox("グラフを選択:", [""] + saved_files, format_func=lambda x: "選択してください" if x == "" else x)
    if st.sidebar.button("グラフを読み込み", disabled=not selected_file):
        graph = load_graph_from_json(selected_file)
        if graph:
            st.session_state.prop_log_graph = graph
            st.session_state.prop_log_graph_name = selected_file
            # グラフを切り替えたら、すべての結果をリセット
            st.session_state.prop_log_latest_results = None
            st.session_state.prop_log_previous_results = None
            st.toast(f"`{selected_file}` を読み込みました。", icon="✅")
            st.rerun()

active_graph = st.session_state.get('prop_log_graph')

if active_graph:
    st.sidebar.markdown("---")
    st.sidebar.header("Step 2: シミュレーションを実行")
    
    default_seeds = ""
    if active_graph.number_of_nodes() > 0:
        num_samples = min(3, active_graph.number_of_nodes())
        try:
            nodes = list(active_graph.nodes())
            default_seeds = ", ".join(map(str, random.sample(nodes, num_samples)))
        except ValueError:
            default_seeds = ""

    seed_nodes_input = st.sidebar.text_input("シードノード (カンマ区切り)", value=default_seeds, key="prop_log_seed_input")
    
    parsed_seeds = []
    if seed_nodes_input.strip():
        try:
            potential_seeds = [int(s.strip()) for s in seed_nodes_input.split(',') if s.strip()]
            parsed_seeds = [s for s in potential_seeds if s in active_graph.nodes()]
        except ValueError:
            st.sidebar.error("シードノードは数値で入力してください。")

    if st.sidebar.button("伝播シミュレーション実行", disabled=not parsed_seeds):
        # 実行前に、現在の「最新結果」を「前回の結果」に移動
        if st.session_state.prop_log_latest_results:
            st.session_state.prop_log_previous_results = st.session_state.prop_log_latest_results.copy()

        # シミュレーション実行
        final_nodes, log = simulate_ic(active_graph.copy(), set(parsed_seeds))
        
        # 結果を計算
        cumulative_activated = {0: set(parsed_seeds)}
        if log:
            log_df = pd.DataFrame(log)
            max_step = int(log_df['step'].max()) if not log_df.empty else 0
            current_total = set(parsed_seeds)
            for i in range(1, max_step + 1):
                newly_activated = set(log_df[log_df['step'] == i]['target'])
                current_total.update(newly_activated)
                cumulative_activated[i] = current_total.copy()
        
        # 新しい結果を「最新結果」として保存
        st.session_state.prop_log_latest_results = {
            "seeds": set(parsed_seeds),
            "log": log,
            "final_activated": final_nodes,
            "cumulative_activated": cumulative_activated
        }
        st.rerun()

# --- メインエリア ---
st.title("影響伝播シミュレーション (比較表示)")

if not active_graph:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"対象グラフ: `{st.session_state.prop_log_graph_name}`")
st.metric("ノード数", active_graph.number_of_nodes())

st.markdown("---")

# --- 結果表示用のヘルパー関数 ---
def display_simplified_results(results, title="前回の結果"):
    """メトリクスと最終ノードリストのみのシンプルな結果表示"""
    with st.container(border=True):
        st.subheader(title)
        st.metric("最終的な活性化ノード総数", len(results['final_activated']))
        st.write("**初期シードノード:**")
        st.write(f"`{sorted(list(results['seeds']))}`")
        st.write("**最終的に活性化した全ノード:**")
        st.text_area("Final Nodes", value=", ".join(map(str, sorted(list(results['final_activated'])))), height=100, key=f"textarea_{title}")

def display_full_results(results, graph, title="最新の結果"):
    """スライダーやグラフを含む詳細な結果表示"""
    with st.container(border=True):
        st.subheader(title)
        st.metric("最終的な活性化ノード総数", len(results['final_activated']))
        
        cumulative_map = results['cumulative_activated']
        max_step = max(cumulative_map.keys()) if cumulative_map else 0
        
        chosen_step = 0
        if max_step > 0:
            chosen_step = st.slider("表示するステップを選択:", 0, max_step, max_step, key=f"slider_{title}")

        st.write(f"**ステップ {chosen_step} での活性化ノード:**")
        active_at_step = cumulative_map.get(chosen_step, set())
        st.info(f"総数: {len(active_at_step)} | `{sorted(list(active_at_step))}`")

        # グラフ可視化
        vis_nodes = []
        seeds = results['seeds']
        for node in graph.nodes():
            color, size, shape = "#D3D3D3", 12, "dot"
            if node in active_at_step:
                if node in seeds: color, size, shape = "red", 25, "star"
                else: color, size = "orange", 18
            vis_nodes.append(Node(id=str(node), label=str(node), color=color, size=size, shape=shape))

        vis_edges = []
        log_df = pd.DataFrame(results['log'])
        for u, v, d in graph.edges(data=True):
            color, width = "#E0E0E0", 1.0
            is_used = not log_df[(log_df['source'] == u) & (log_df['target'] == v) & (log_df['step'] <= chosen_step)].empty
            if is_used: color, width = "blue", 2.5
            elif u in active_at_step and v not in active_at_step: color = "#FFC0CB"
            vis_edges.append(Edge(source=str(u), target=str(v), color=color, width=width, label=f"{d.get('weight',0):.2f}"))

        config = Config(width="100%", height=500, directed=True, physics=False)
        agraph(nodes=vis_nodes, edges=vis_edges, config=config)

# --- 結果の表示ロジック ---
latest_results = st.session_state.get('prop_log_latest_results')
previous_results = st.session_state.get('prop_log_previous_results')

if latest_results and previous_results:
    # 2つの結果を比較表示
    col1, col2 = st.columns(2)
    with col1:
        display_simplified_results(previous_results, title="前回の結果")
    with col2:
        display_full_results(latest_results, active_graph, title="最新の結果")
elif latest_results:
    # 最新の結果のみ表示
    display_full_results(latest_results, active_graph, title="最新の結果")
else:
    # シミュレーション未実行
    st.info("サイドバーでシードノードを設定し、「伝播シミュレーション実行」ボタンを押してください。")