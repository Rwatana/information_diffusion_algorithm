import streamlit as st
import pandas as pd
import sys
import os
import json
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config


# --- パス設定とモジュールのインポート ---
try:
    # スクリプトの実行ディレクトリに基づいてプロジェクトルートを動的に設定
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 影響拡散シミュレーション用の関数をインポート
    from datagen.data_utils import simulate_ic
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"必要なモジュールの読み込みに失敗しました: {e}")
    st.info(
        "このアプリを実行するには、プロジェクトのルートディレクトリに `datagen/data_utils.py` が必要です。"
    )
    st.stop()


# --- 定数とディレクトリ設定 ---
SAVE_DIR_NAME = "saved_graphs"
# このスクリプトと同じ階層にある`saved_graphs`を保存場所とする
SAVE_DIR_PATH = os.path.join(current_file_dir, SAVE_DIR_NAME)
if not os.path.exists(SAVE_DIR_PATH):
    os.makedirs(SAVE_DIR_PATH)
    st.warning(
        f"保存ディレクトリを作成しました: `{SAVE_DIR_PATH}`"
    )


# --- ヘルパー関数 ---
def load_graph_from_json(folder_name):
    """フォルダ名を受け取り、その中のgraph_data.jsonを読み込み、networkxグラフを返します。"""
    filepath = os.path.join(SAVE_DIR_PATH, folder_name, "graph_data.json")
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    except Exception as e:
        st.error(f"グラフの読み込み中にエラーが発生しました: {e}")
        return None


def get_saved_graph_files():
    """保存ディレクトリからグラフのフォルダリストを取得します。"""
    if not os.path.exists(SAVE_DIR_PATH):
        return []
    # ディレクトリのみをリストアップし、降順（新しいものが上）でソート
    return sorted(
        [
            d
            for d in os.listdir(SAVE_DIR_PATH)
            if os.path.isdir(os.path.join(SAVE_DIR_PATH, d))
        ],
        reverse=True,
    )


def select_seeds_by_greedy(G, k, num_simulations=10, progress_placeholder=None):
    """
    Greedyアルゴリズムを用いて影響を最大化するk個のシードノードを選択します。
    マージナルゲインの計算は、指定された回数のモンテカルロシミュレーションによって推定されます。

    Args:
        G (nx.Graph): グラフ。
        k (int): 選択するシード数。
        num_simulations (int): 影響を推定するためのシミュレーション回数。
        progress_placeholder (st.empty): Streamlitのプログレスバーを表示するプレースホルダー。

    Returns:
        list: 選択されたシードノードのリスト。
    """
    seeds = []
    current_influence = 0.0

    # 全体の進捗状況を計算するための準備
    candidate_nodes_master = list(G.nodes())
    total_iterations = sum(range(len(candidate_nodes_master), len(candidate_nodes_master) - k, -1))
    iterations_done = 0

    for i in range(k):
        best_node = -1
        best_marginal_gain = -1.0

        candidate_nodes = [node for node in candidate_nodes_master if node not in seeds]

        for j, node in enumerate(candidate_nodes):
            iterations_done += 1
            if progress_placeholder:
                progress_value = iterations_done / total_iterations if total_iterations > 0 else 0
                progress_text = f"シード {i+1}/{k} を探索中... (候補: {node})"
                progress_placeholder.progress(progress_value, text=progress_text)

            # 候補ノードを追加した場合の影響を推定
            total_spread = 0
            for _ in range(num_simulations):
                activated_nodes, _ = simulate_ic(G.copy(), seeds + [node])
                total_spread += len(activated_nodes)

            avg_spread = total_spread / num_simulations
            marginal_gain = avg_spread - current_influence

            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_node = node

        if best_node != -1:
            seeds.append(best_node)
            # 新しいシードセットでの影響を再計算して更新
            total_spread_with_new_seed = 0
            for _ in range(num_simulations):
                activated_nodes, _ = simulate_ic(G.copy(), seeds)
                total_spread_with_new_seed += len(activated_nodes)
            current_influence = total_spread_with_new_seed / num_simulations
        else:
            # どのノードを追加しても影響が増加しない場合は早期終了
            st.warning("マージナルゲインが0になったため、シード選択を早期終了しました。")
            break

    if progress_placeholder:
        progress_placeholder.empty()

    return seeds


# --- セッションステートの初期化 ---
# このページ専用のキーを使い、他のページとの競合を避ける
if "greedy_graph" not in st.session_state:
    st.session_state.greedy_graph = None
if "greedy_graph_name" not in st.session_state:
    st.session_state.greedy_graph_name = "未選択"
if "greedy_simulation_results" not in st.session_state:
    st.session_state.greedy_simulation_results = None


# --- サイドバー ---
st.sidebar.title("Greedy Simulation")
st.sidebar.header("Step 1: グラフを選択")

saved_files = get_saved_graph_files()
if not saved_files:
    st.sidebar.warning("読み込み可能なグラフがありません。`1_graph_visualization.py`でグラフを作成・保存してください。")
else:
    # `format_func`で見栄えを良くする
    selected_file = st.sidebar.selectbox(
        "グラフを選択:",
        [""] + saved_files,
        format_func=lambda x: "ファイルを選択" if x == "" else x,
        key="greedy_load_selector",
    )
    if st.sidebar.button("グラフを読み込み", disabled=not selected_file):
        graph = load_graph_from_json(selected_file)
        if graph:
            st.session_state.greedy_graph = graph
            st.session_state.greedy_graph_name = selected_file
            st.session_state.greedy_simulation_results = None  # 新しいグラフを読んだら結果をリセット
            st.toast(f"`{selected_file}` を読み込みました。", icon="✅")
            st.rerun() # 画面を再描画して状態を反映

# --- メインエリア ---
st.title("Greedy アルゴリズムベースの影響最大化シミュレーション")

G = st.session_state.get("greedy_graph")

if G is None:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"対象グラフ: `{st.session_state.greedy_graph_name}`")
st.metric("ノード数", G.number_of_nodes())
st.markdown("---")

# --- シミュレーション設定（グラフ読み込み後に表示） ---
st.sidebar.markdown("---")
st.sidebar.header("Step 2: シミュレーションを実行")

max_seeds = G.number_of_nodes()
num_seeds = st.sidebar.slider(
    "シード数 (k)", 1, max_seeds, min(5, max_seeds), key="greedy_num_seeds"
)

num_sims_for_greedy = st.sidebar.slider(
    "Greedy用シミュレーション回数（精度）",
    min_value=5,
    max_value=100,
    value=10,
    step=5,
    key="greedy_num_sims",
    help="Greedyアルゴリズムが各ステップでノードの良さを評価する際のシミュレーション回数です。大きいほど正確になりますが、計算時間が大幅に増加します。",
)

if st.sidebar.button("Greedyでシミュレーション実行", key="greedy_run_sim_button"):
    st.session_state.greedy_simulation_results = None  # 実行のたびに結果をクリア

    # プログレスバーを表示するためのプレースホルダー
    progress_placeholder = st.empty()

    with st.spinner("Greedyアルゴリズムでシードを選択中...（時間がかかる場合があります）"):
        seeds = select_seeds_by_greedy(G, num_seeds, num_sims_for_greedy, progress_placeholder)

    if seeds:
        st.toast(f"Greedyアルゴリズムにより {len(seeds)}個のシードを選択しました。", icon="🏆")
        with st.spinner("最終的な伝播シミュレーションを実行中..."):
            # 最終的な結果を得るために、選択されたシードで一度シミュレーションを実行
            final_activated_nodes, raw_log = simulate_ic(G.copy(), seeds)

        # ステップごとの累積活性化ノードを計算
        stepwise_cumulative = {0: set(seeds)}
        if raw_log:
            df_log = pd.DataFrame(raw_log)
            # `step`列が空でないことを確認
            if not df_log.empty and 'step' in df_log.columns:
                max_step = int(df_log["step"].max())
                current_cumulative = set(seeds)
                for step in range(1, max_step + 1):
                    newly_activated = set(df_log[df_log["step"] == step]["target"].unique())
                    current_cumulative.update(newly_activated)
                    stepwise_cumulative[step] = current_cumulative.copy()

        st.session_state.greedy_simulation_results = {
            "seeds": seeds,
            "log": raw_log,
            "final_activated": final_activated_nodes,
            "cumulative": stepwise_cumulative,
        }
        st.rerun()
    else:
        st.error("Greedyアルゴリズムでシードを選択できませんでした。")


# --- 結果表示 ---
if st.session_state.get("greedy_simulation_results"):
    results = st.session_state.greedy_simulation_results
    st.header("シミュレーション結果")

    res_cols = st.columns(2)
    res_cols[0].metric("選択されたシード数", len(results["seeds"]))
    res_cols[1].metric("最終的な活性化ノード数", len(results["final_activated"]))
    st.info(f"選択されたシード (Greedy): `{sorted(list(results['seeds']))}`")

    st.subheader("ステップごとのグラフ状態可視化")
    cumulative_map = results["cumulative"]
    max_slider_step = max(cumulative_map.keys()) if cumulative_map else 0

    selected_step = 0
    if max_slider_step > 0:
        selected_step = st.slider(
            "表示ステップ選択",
            0,
            max_slider_step,
            max_slider_step, # デフォルトで最終ステップを表示
            key="greedy_step_slider_viz",
        )

    # 選択されたステップでの活性化ノードセットを取得
    nodes_active_now = cumulative_map.get(selected_step, set())

    # 可視化用のノードとエッジのリストを作成
    nodes_v, edges_v = [], []
    for node_id in G.nodes():
        color, size, shape = "#D3D3D3", 12, "dot" # 未活性ノードは薄いグレー
        if node_id in nodes_active_now:
            if node_id in results["seeds"]:
                color, size, shape = "#FF4B4B", 25, "star" # シードは赤色の星
            else:
                color, size = "#FFA500", 18 # 活性化済みはオレンジ
        nodes_v.append(
            Node(
                id=str(node_id), label=str(node_id), color=color, size=size, shape=shape
            )
        )
    
    # ログからエッジの状態を決定
    log_df = pd.DataFrame(results["log"])
    for u, v, data in G.edges(data=True):
        ec, ew = "#E0E0E0", 1 # デフォルトのエッジ色と太さ
        # 選択されたステップまでにこのエッジが伝播に使われたか
        is_used = not log_df[
            (log_df["source"] == u)
            & (log_df["target"] == v)
            & (log_df["step"] <= selected_step)
        ].empty if not log_df.empty else False

        if is_used:
            ec, ew = "#0000FF", 2.5 # 伝播成功エッジは青く太く
        elif u in nodes_active_now and v in nodes_active_now:
            ec = "#ADD8E6" # 両端が活性化しているが伝播には使われなかったエッジ

        edges_v.append(
            Edge(
                source=str(u),
                target=str(v),
                color=ec,
                width=ew,
                label=f"{data.get('weight', 0):.2f}", # エッジの重みをラベル表示
            )
        )

    # グラフの描画設定
    config_viz = Config(
        width="100%", height=700, directed=G.is_directed(), physics=False
    )
    st.write(f"**ステップ {selected_step} の状態:**")
    st.caption(
        "凡例 - ノード[赤(星): シード, オレンジ: 活性化, グレー: 未活性] | エッジ[青: 伝播成功, 水色: 両端活性(非伝播), グレー: 未使用]"
    )
    # agraphコンポーネントでグラフを描画
    agraph(nodes=nodes_v, edges=edges_v, config=config_viz)
