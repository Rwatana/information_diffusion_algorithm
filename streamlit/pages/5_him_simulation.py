import streamlit as st
import pandas as pd
import sys
import os
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import torch  # HIMモデルがPyTorchを使用

# --- パス設定とモジュールのインポート ---
try:
    # このスクリプト(5_him_simulation.py)の場所を基準にプロジェクトルートを特定
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 必要なモジュールをインポート
    from datagen.data_utils import simulate_ic, generate_propagations
    from him_full.him_model import HIMModel
    from him_full.seed_selection import adaptive_sliding_window
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"必要なモジュールの読み込みに失敗しました: {e}")
    st.info(
        "プロジェクトのディレクトリ構造 (`datagen`, `him_full` フォルダ) が正しいか確認してください。"
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
if "him_graph" not in st.session_state:
    st.session_state.him_graph = None
if "him_graph_name" not in st.session_state:
    st.session_state.him_graph_name = "未選択"
if "him_simulation_results" not in st.session_state:
    st.session_state.him_simulation_results = None


# --- サイドバー ---
st.sidebar.title("HIM Simulation")
st.sidebar.header("Step 1: グラフを選択")

saved_files = get_saved_graph_files()
if not saved_files:
    st.sidebar.error("読み込み可能なグラフがありません。")
else:
    selected_file = st.sidebar.selectbox(
        "グラフを選択:",
        [""] + saved_files,
        format_func=lambda x: "ファイルを選択" if x == "" else x,
        key="him_load_selector",
    )
    if st.sidebar.button("グラフを読み込み", disabled=not selected_file):
        graph = load_graph_from_json(selected_file)
        if graph:
            st.session_state.him_graph = graph
            st.session_state.him_graph_name = selected_file
            st.session_state.him_simulation_results = None  # 結果をリセット
            st.toast(f"`{selected_file}` を読み込みました。", icon="✅")
            st.rerun()

# --- メインエリア ---
st.title("HIM ベースの影響最大化シミュレーション")

G = st.session_state.get("him_graph")

if G is None:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"対象グラフ: `{st.session_state.him_graph_name}`")
st.metric("ノード数", G.number_of_nodes())
st.markdown("---")

# --- シミュレーション設定（グラフ読み込み後に表示） ---
st.sidebar.markdown("---")
st.sidebar.header("Step 2: HIM & シミュレーション設定")

max_seeds = G.number_of_nodes()
num_seeds = st.sidebar.slider(
    "シード数 (k)", 1, max_seeds, min(10, max_seeds), key="him_num_seeds"
)

st.sidebar.subheader("HIMモデル学習設定")
him_dim = st.sidebar.slider(
    "埋め込み次元 (dim)", 8, 128, 32, step=8, key="him_dim_param"
)
him_epochs = st.sidebar.slider("学習エポック数", 1, 500, 10, key="him_epochs_param")
him_beta = st.sidebar.slider(
    "Adaptive Sliding Window Beta", 0.1, 5.0, 1.0, step=0.1, key="him_beta_param"
)

generate_dummy_props = st.sidebar.checkbox(
    "簡易伝播インスタンスを学習に使用する", value=True, key="him_gen_props"
)
num_dummy_prop_instances = 0
if generate_dummy_props:
    num_dummy_prop_instances = st.sidebar.slider(
        "簡易伝播インスタンス数", 1, 100, 10, key="him_num_dummy_props"
    )


if st.sidebar.button("HIMモデル学習 & シミュレーション実行", key="him_run_sim_button"):
    st.session_state.him_simulation_results = None

    # 1. (オプション) 簡易伝播インスタンスの生成
    propagations = []
    if generate_dummy_props and num_dummy_prop_instances > 0:
        with st.spinner(
            f"{num_dummy_prop_instances}件の簡易伝播インスタンスを生成中..."
        ):
            prop_seed_count = max(1, min(5, G.number_of_nodes() // 10))
            propagations = generate_propagations(
                G, seed_count=prop_seed_count, num_instances=num_dummy_prop_instances
            )
            st.write(f"{len(propagations)}件の伝播インスタンスを学習に使用します。")

    # 2. HIMモデルの学習
    seeds = None
    with st.spinner(f"HIMモデルを学習中 (dim={him_dim}, epochs={him_epochs})..."):
        try:
            model = HIMModel(num_nodes=G.number_of_nodes(), dim=him_dim)
            model.fit(G, propagations, epochs=him_epochs, verbose=False)
            embeddings = model.embeddings.detach().cpu()
            st.success("HIMモデルの学習が完了しました。")

            # 3. シード選択
            with st.spinner("学習済み埋め込みを使用してシードを選択中..."):
                seeds = adaptive_sliding_window(
                    G, embeddings, k=num_seeds, beta=him_beta
                )
                st.success("シード選択が完了しました。")

        except Exception as e:
            st.error(f"HIMモデルの処理中にエラーが発生しました: {e}")

    # 4. 伝播シミュレーション
    if seeds:
        st.toast(f"HIMにより {len(seeds)}個のシードを選択しました。")
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

        st.session_state.him_simulation_results = {
            "seeds": seeds,
            "log": raw_log,
            "final_activated": final_activated_nodes,
            "cumulative": stepwise_cumulative,
        }
        st.rerun()
    elif st.session_state.get("him_run_sim_button"):
        st.error(
            "HIMによるシード選択に失敗したため、シミュレーションを実行できません。"
        )


# --- 結果表示 ---
if st.session_state.get("him_simulation_results"):
    results = st.session_state.him_simulation_results
    st.header("シミュレーション結果 (HIM)")

    res_cols = st.columns(2)
    res_cols[0].metric("選択されたシード数", len(results["seeds"]))
    res_cols[1].metric("最終的な活性化ノード数", len(results["final_activated"]))
    st.info(f"選択されたシード (HIM): `{sorted(list(results['seeds']))}`")

    # --- ここから新規追加：ステップごとの新規活性化ノード表示 ---
    st.subheader("ステップごとの新規活性化ノード")
    log_df = pd.DataFrame(results["log"])
    initial_seeds = results["seeds"]

    newly_activated_per_step = {0: sorted(list(initial_seeds))}
    all_activated_so_far = set(initial_seeds)

    if not log_df.empty:
        max_step_disp = int(log_df["step"].max())
        for step in range(1, max_step_disp + 1):
            targets_this_step = set(log_df[log_df["step"] == step]["target"])
            truly_new_nodes = targets_this_step - all_activated_so_far
            if truly_new_nodes:
                newly_activated_per_step[step] = sorted(list(truly_new_nodes))
            all_activated_so_far.update(targets_this_step)

    # データフレーム表示用の整形
    df_display_data = {}
    if newly_activated_per_step:
        max_len = max(
            len(nodes) for nodes in newly_activated_per_step.values() if nodes
        )
        for step, nodes in newly_activated_per_step.items():
            padded_nodes = nodes + [""] * (max_len - len(nodes))
            df_display_data[f"Step {step}"] = padded_nodes

    if df_display_data:
        st.dataframe(pd.DataFrame(df_display_data), use_container_width=True)
    else:
        st.write("新たな活性化はありませんでした。")
    # --- ここまで新規追加 ---

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
            key="him_step_slider_viz",
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
