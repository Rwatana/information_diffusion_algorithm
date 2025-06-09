import streamlit as st
import pandas as pd
import sys
import os
import json
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

# --- パス設定とモジュールのインポート ---
try:
    # このスクリプト(4_degree_simulation.py)の場所を基準にプロジェクトルートを特定
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 必要なモジュールをインポート
    from datagen.data_utils import simulate_ic
    from degree_centrality.degree_centrality import select_seeds_by_degree_centrality
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"必要なモジュールの読み込みに失敗しました: {e}")
    st.info(
        "プロジェクトのディレクトリ構造 (`datagen`, `degree_centrality` フォルダ) が正しいか確認してください。"
    )
    st.stop()


# --- 定数とディレクトリ設定 ---
SAVE_DIR_NAME = "saved_graphs"
# このスクリプトと同じ階層にある`saved_graphs`を保存場所とする
SAVE_DIR_PATH = os.path.join(current_file_dir, SAVE_DIR_NAME)
if not os.path.exists(SAVE_DIR_PATH):
    st.error(f"保存ディレクトリが見つかりません: {SAVE_DIR_PATH}")
    st.info(
        "`1_graph_visualization.py`ページでグラフを保存すると、自動的に作成されます。"
    )
    st.stop()


# --- ヘルパー関数 ---
def load_graph_from_json(folder_name):
    """フォルダ名を受け取り、その中のgraph_data.jsonを読み込みます。"""
    filepath = os.path.join(SAVE_DIR_PATH, folder_name, "graph_data.json")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    except Exception as e:
        st.error(f"グラフ読込エラー: {e}")
        return None


def get_saved_graph_files():
    """保存されているグラフの「フォルダ」リストを取得します。"""
    if not os.path.exists(SAVE_DIR_PATH):
        return []
    return sorted(
        [
            d
            for d in os.listdir(SAVE_DIR_PATH)
            if os.path.isdir(os.path.join(SAVE_DIR_PATH, d))
        ],
        reverse=True,
    )


# --- セッションステートの初期化 ---
# このページ専用のキーを使用
if "dc_graph" not in st.session_state:
    st.session_state.dc_graph = None
if "dc_graph_name" not in st.session_state:
    st.session_state.dc_graph_name = "未選択"
if "dc_simulation_results" not in st.session_state:
    st.session_state.dc_simulation_results = None


# --- サイドバー ---
st.sidebar.title("Degree Centrality Simulation")
st.sidebar.header("Step 1: グラフを選択")

saved_files = get_saved_graph_files()
if not saved_files:
    st.sidebar.error("読み込み可能なグラフがありません。")
else:
    selected_file = st.sidebar.selectbox(
        "グラフを選択:",
        [""] + saved_files,
        format_func=lambda x: "ファイルを選択" if x == "" else x,
        key="dc_load_selector",
    )
    if st.sidebar.button("グラフを読み込み", disabled=not selected_file):
        graph = load_graph_from_json(selected_file)
        if graph:
            st.session_state.dc_graph = graph
            st.session_state.dc_graph_name = selected_file
            st.session_state.dc_simulation_results = None  # 結果をリセット
            st.toast(f"`{selected_file}` を読み込みました。", icon="✅")
            st.rerun()

# --- メインエリア ---
st.title("次数中心性 ベースの影響最大化シミュレーション")

G = st.session_state.get("dc_graph")

if G is None:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"対象グラフ: `{st.session_state.dc_graph_name}`")
st.metric("ノード数", G.number_of_nodes())
st.markdown("---")

# --- シミュレーション設定（グラフ読み込み後に表示） ---
st.sidebar.markdown("---")
st.sidebar.header("Step 2: シミュレーションを実行")

degree_type_option = st.sidebar.selectbox(
    "次数タイプ", ["Out-Degree", "In-Degree"], key="dc_degree_type"
)
use_out_degree_flag = True if degree_type_option == "Out-Degree" else False

max_seeds = G.number_of_nodes()
num_seeds = st.sidebar.slider(
    "シード数 (k)", 1, max_seeds, min(10, max_seeds), key="dc_num_seeds"
)

if st.sidebar.button(
    f"{degree_type_option}でシミュレーション実行", key="dc_run_sim_button"
):
    st.session_state.dc_simulation_results = None  # 実行のたびに結果をクリア

    with st.spinner(f"次数中心性 ({degree_type_option}) でシードを選択中..."):
        seeds = select_seeds_by_degree_centrality(
            G, num_seeds, use_out_degree=use_out_degree_flag
        )

    if seeds:
        st.toast(f"{degree_type_option}により {len(seeds)}個のシードを選択しました。")
        with st.spinner("伝播シミュレーションを実行中..."):
            final_activated_nodes, raw_log = simulate_ic(G.copy(), seeds)

        stepwise_cumulative = {0: set(seeds)}
        if raw_log:
            df_log = pd.DataFrame(raw_log)
            max_step = int(df_log["step"].max()) if not df_log.empty else 0
            current_cumulative = set(seeds)
            for step in range(1, max_step + 1):
                newly_activated = set(df_log[df_log["step"] == step]["target"].unique())
                current_cumulative.update(newly_activated)
                stepwise_cumulative[step] = current_cumulative.copy()

        st.session_state.dc_simulation_results = {
            "degree_type": degree_type_option,  # 実行時の次数タイプを保存
            "seeds": seeds,
            "log": raw_log,
            "final_activated": final_activated_nodes,
            "cumulative": stepwise_cumulative,
        }
        st.rerun()
    else:
        st.error(f"次数中心性 ({degree_type_option}) でシードを選択できませんでした。")


# --- 結果表示 ---
if st.session_state.get("dc_simulation_results"):
    results = st.session_state.dc_simulation_results
    run_degree_type = results["degree_type"]  # 保存した次数タイプを取得

    st.header(f"シミュレーション結果 ({run_degree_type})")

    res_cols = st.columns(2)
    res_cols[0].metric("選択されたシード数", len(results["seeds"]))
    res_cols[1].metric("最終的な活性化ノード数", len(results["final_activated"]))
    st.info(f"選択されたシード ({run_degree_type}): `{sorted(list(results['seeds']))}`")

    st.subheader("ステップごとのグラフ状態可視化")
    cumulative_map = results["cumulative"]
    max_slider_step = max(cumulative_map.keys()) if cumulative_map else 0

    selected_step = 0
    if max_slider_step > 0:
        selected_step = st.slider(
            "表示ステップ選択",
            0,
            max_slider_step,
            max_slider_step,
            key="dc_step_slider_viz",
        )

    nodes_active_now = cumulative_map.get(selected_step, set())

    nodes_v, edges_v = [], []
    for node_id in G.nodes():
        color, size, shape = "#E0E0E0", 12, "dot"
        if node_id in nodes_active_now:
            if node_id in results["seeds"]:
                color, size, shape = "red", 25, "star"
            else:
                color, size = "orange", 18
        nodes_v.append(
            Node(
                id=str(node_id), label=str(node_id), color=color, size=size, shape=shape
            )
        )

    log_df = pd.DataFrame(results["log"])
    for u, v, data in G.edges(data=True):
        ec, ew = "#E0E0E0", 1
        is_used = not log_df[
            (log_df["source"] == u)
            & (log_df["target"] == v)
            & (log_df["step"] <= selected_step)
        ].empty
        if is_used:
            ec, ew = "blue", 2.5
        elif u in nodes_active_now and v in nodes_active_now:
            ec = "#B0C4DE"
        edges_v.append(
            Edge(
                source=str(u),
                target=str(v),
                color=ec,
                width=ew,
                label=f"{data.get('weight',0):.2f}",
            )
        )

    config_viz = Config(
        width="100%", height=700, directed=G.is_directed(), physics=True
    )
    st.write(f"**ステップ {selected_step} の状態:**")
    st.caption(
        "ノード色 - 赤(星): 初期シード, オレンジ: 活性化済み, グレー: 未活性 | エッジ色 - 青: 伝播成功, 水色: 両端活性(非伝播)"
    )
    agraph(nodes=nodes_v, edges=edges_v, config=config_viz)
