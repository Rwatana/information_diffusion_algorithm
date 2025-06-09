import streamlit as st
import pandas as pd
import sys
import os
import json
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config


# --- パス設定とモジュールのインポート ---
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from datagen.data_utils import simulate_ic
    from pagerank.pagerank import select_seeds_by_pagerank
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"必要なモジュールの読み込みに失敗しました: {e}")
    st.info("プロジェクトのディレクトリ構造が正しいか、必要なファイルが存在するか確認してください。")
    st.stop()


# --- 定数とディレクトリ設定 ---
SAVE_DIR_NAME = "saved_graphs"
# このスクリプトと同じ階層にある`saved_graphs`を保存場所とする
SAVE_DIR_PATH = os.path.join(current_file_dir, SAVE_DIR_NAME)
if not os.path.exists(SAVE_DIR_PATH):
    st.error(f"保存ディレクトリが見つかりません: {SAVE_DIR_PATH}")
    st.info("`1_graph_visualization.py`ページでグラフを保存すると、自動的に作成されます。")
    st.stop()


# --- ヘルパー関数 ---
def load_graph_from_json(folder_name):
    """フォルダ名を受け取り、その中のgraph_data.jsonを読み込みます。"""
    filepath = os.path.join(SAVE_DIR_PATH, folder_name, 'graph_data.json')
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    except Exception as e:
        st.error(f"グラフ読込エラー: {e}")
        return None

def get_saved_graph_files():
    """保存されているグラフの「フォルダ」リストを取得します。"""
    if not os.path.exists(SAVE_DIR_PATH): return []
    return sorted([d for d in os.listdir(SAVE_DIR_PATH) if os.path.isdir(os.path.join(SAVE_DIR_PATH, d))], reverse=True)


# --- セッションステートの初期化 ---
# このページ専用のキーを使用
if 'pr_graph' not in st.session_state:
    st.session_state.pr_graph = None
if 'pr_graph_name' not in st.session_state:
    st.session_state.pr_graph_name = "未選択"
if 'pr_simulation_results' not in st.session_state:
    st.session_state.pr_simulation_results = None


# --- サイドバー ---
st.sidebar.title("PageRank Simulation")
st.sidebar.header("Step 1: グラフを選択")

saved_files = get_saved_graph_files()
if not saved_files:
    st.sidebar.error("読み込み可能なグラフがありません。")
else:
    selected_file = st.sidebar.selectbox(
        "グラフを選択:", 
        [""] + saved_files, 
        format_func=lambda x: "ファイルを選択" if x == "" else x,
        key="pr_load_selector"
    )
    if st.sidebar.button("グラフを読み込み", disabled=not selected_file):
        graph = load_graph_from_json(selected_file)
        if graph:
            st.session_state.pr_graph = graph
            st.session_state.pr_graph_name = selected_file
            st.session_state.pr_simulation_results = None # 結果をリセット
            st.toast(f"`{selected_file}` を読み込みました。", icon="✅")
            st.rerun()

# --- メインエリア ---
st.title("PageRank ベースの影響最大化シミュレーション")

G = st.session_state.get('pr_graph')

if G is None:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"対象グラフ: `{st.session_state.pr_graph_name}`")
st.metric("ノード数", G.number_of_nodes())
st.markdown("---")

# --- シミュレーション設定（グラフ読み込み後に表示） ---
st.sidebar.markdown("---")
st.sidebar.header("Step 2: シミュレーションを実行")

max_seeds = G.number_of_nodes()
num_seeds = st.sidebar.slider("シード数 (k)", 1, max_seeds, min(10, max_seeds), key="pr_num_seeds")
# propagation_prob_sim = st.sidebar.slider("伝播確率 (p) for Simulation", 0.01, 1.0, 0.1, 0.01, key="pr_prop_prob") # グラフの重みを使うためコメントアウト

if st.sidebar.button("PageRankでシミュレーション実行", key="pr_run_sim_button"):
    st.session_state.pr_simulation_results = None # 実行のたびに結果をクリア

    with st.spinner("PageRank でシードを選択中..."):
        # PageRankでは重み付きグラフを考慮する場合があるため、そのまま渡す
        seeds = select_seeds_by_pagerank(G, num_seeds)
    
    if seeds:
        st.toast(f"PageRankにより {len(seeds)}個のシードを選択しました。")
        with st.spinner("伝播シミュレーションを実行中..."):
            # シミュレーションにはグラフの重み('weight')が使われる
            final_activated_nodes, raw_log = simulate_ic(G.copy(), seeds)
        
        stepwise_cumulative = {0: set(seeds)}
        if raw_log:
            df_log = pd.DataFrame(raw_log)
            max_step = int(df_log['step'].max()) if not df_log.empty else 0
            current_cumulative = set(seeds)
            for step in range(1, max_step + 1):
                newly_activated = set(df_log[df_log['step'] == step]['target'].unique())
                current_cumulative.update(newly_activated)
                stepwise_cumulative[step] = current_cumulative.copy()
        
        st.session_state.pr_simulation_results = {
            "seeds": seeds,
            "log": raw_log,
            "final_activated": final_activated_nodes,
            "cumulative": stepwise_cumulative
        }
        st.rerun()
    else:
        st.error("PageRank でシードを選択できませんでした。")


# --- 結果表示 ---
if st.session_state.get('pr_simulation_results'):
    results = st.session_state.pr_simulation_results
    st.header("シミュレーション結果")
    
    res_cols = st.columns(2)
    res_cols[0].metric("選択されたシード数", len(results['seeds']))
    res_cols[1].metric("最終的な活性化ノード数", len(results['final_activated']))
    st.info(f"選択されたシード (PageRank): `{sorted(list(results['seeds']))}`")

    st.subheader("ステップごとのグラフ状態可視化")
    cumulative_map = results['cumulative']
    max_slider_step = max(cumulative_map.keys()) if cumulative_map else 0
    
    selected_step = 0
    if max_slider_step > 0:
        selected_step = st.slider("表示ステップ選択", 0, max_slider_step, max_slider_step, key="pr_step_slider_viz")
    
    nodes_active_now = cumulative_map.get(selected_step, set())
    
    nodes_v, edges_v = [], []
    for node_id in G.nodes():
        color, size, shape = "#E0E0E0", 12, "dot"
        if node_id in nodes_active_now:
            if node_id in results['seeds']: color, size, shape = "red", 25, "star"
            else: color, size = "orange", 18
        nodes_v.append(Node(id=str(node_id), label=str(node_id), color=color, size=size, shape=shape))

    log_df = pd.DataFrame(results['log'])
    for u, v, data in G.edges(data=True):
        ec, ew = "#E0E0E0", 1
        is_used = not log_df[(log_df['source']==u) & (log_df['target']==v) & (log_df['step'] <= selected_step)].empty
        if is_used: ec, ew = "blue", 2.5
        elif u in nodes_active_now and v in nodes_active_now: ec = "#B0C4DE"
        edges_v.append(Edge(source=str(u), target=str(v), color=ec, width=ew, label=f"{data.get('weight',0):.2f}"))
    
    config_viz = Config(width="100%", height=700, directed=G.is_directed(), physics=True)
    st.write(f"**ステップ {selected_step} の状態:**")
    st.caption("ノード色 - 赤(星): 初期シード, オレンジ: 活性化済み, グレー: 未活性 | エッジ色 - 青: 伝播成功, 水色: 両端活性(非伝播)")
    agraph(nodes=nodes_v, edges=edges_v, config=config_viz)