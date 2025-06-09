import streamlit as st
import networkx as nx
import random
import os
import json
from datetime import datetime
from streamlit_agraph import agraph, Node, Edge, Config

# --- 定数とパス設定 ---
# スクリプトの場所を基準に保存ディレクトリを設定
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(APP_ROOT_DIR, "saved_graphs")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- ヘルパー関数 (グラフの保存と読込) ---


def save_graph_to_json(graph, base_name="graph_export"):
    """
    NetworkXグラフを属性付きでJSONファイルに保存します。
    (ノードリンク形式を使用)
    """
    if not graph:
        st.error("保存対象のグラフがありません。")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    full_folder_path = os.path.join(SAVE_DIR, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    # nx.node_link_data を使って、グラフ構造と属性(重みなど)をまとめて辞書に変換
    graph_data = nx.node_link_data(graph)

    file_path = os.path.join(full_folder_path, "graph_data.json")
    try:
        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)
        return folder_name
    except Exception as e:
        st.error(f"グラフの保存中にエラーが発生しました: {e}")
        return None


def load_graph_from_json(folder_name):
    """
    JSONファイルからNetworkXグラフを復元します。
    """
    file_path = os.path.join(SAVE_DIR, folder_name, "graph_data.json")
    try:
        with open(file_path, "r") as f:
            graph_data = json.load(f)
        # nx.node_link_graph を使って、辞書からグラフオブジェクトを復元
        return nx.node_link_graph(graph_data)
    except FileNotFoundError:
        st.error(f"ファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        st.error(f"グラフの読込中にエラーが発生しました: {e}")
        return None


def get_saved_experiments():
    """
    保存されているグラフのフォルダリストを取得します。
    """
    if not os.path.exists(SAVE_DIR):
        return []
    # graph_data.json を含むディレクトリのみをリストアップ
    return sorted(
        [
            d
            for d in os.listdir(SAVE_DIR)
            if os.path.isdir(os.path.join(SAVE_DIR, d))
            and os.path.exists(os.path.join(SAVE_DIR, d, "graph_data.json"))
        ],
        reverse=True,
    )


# --- Streamlit アプリケーションのUI ---

st.set_page_config(layout="wide", page_title="グラフ可視化 & 確率設定")
st.title("グラフ可視化 & 伝播確率設定")

# --- セッションステートの初期化 ---
if "graph" not in st.session_state:
    st.session_state.graph = None  # 現在のグラフオブジェクトを保持
if "graph_name" not in st.session_state:
    st.session_state.graph_name = "未生成"  # 表示用のグラフ名

# --- サイドバー ---

# --- 1. グラフ生成セクション ---
st.sidebar.header("グラフ生成")

num_nodes = st.sidebar.slider("ノード数", 5, 100, 20, key="num_nodes")
edge_density = st.sidebar.slider("辺密度", 0.01, 0.5, 0.1, key="edge_density")
is_directed = st.sidebar.checkbox("有向グラフ", True, key="is_directed")

st.sidebar.subheader("伝播確率の設定")
prob_min = st.sidebar.slider("最小伝播確率", 0.0, 1.0, 0.01, 0.01, key="prob_min")
prob_max = st.sidebar.slider("最大伝播確率", 0.0, 1.0, 0.2, 0.01, key="prob_max")

# 最小値 > 最大値 の場合はボタンを無効化
generate_disabled = False
if prob_min > prob_max:
    st.sidebar.error("最小確率は最大確率以下に設定してください。")
    generate_disabled = True

if st.sidebar.button(
    "新しいグラフを生成 (確率付き)", disabled=generate_disabled, key="generate_graph"
):
    # Erdos-Renyiモデルで基本グラフを生成
    base_graph = nx.gnp_random_graph(num_nodes, edge_density, directed=is_directed)

    # 各エッジに指定範囲のランダムな伝播確率(weight)を割り当て
    for u, v in base_graph.edges():
        base_graph[u][v]["weight"] = random.uniform(prob_min, prob_max)

    # セッションステートに保存
    st.session_state.graph = base_graph
    st.session_state.graph_name = (
        f"新規生成 ({num_nodes}ノード, {base_graph.number_of_edges()}エッジ)"
    )
    st.sidebar.success("新しいグラフを生成しました。")
    st.rerun()  # 画面を更新して即座にグラフを表示

st.sidebar.markdown("---")


# --- 2. 保存・読込セクション ---
st.sidebar.header("保存 & 読込")

# 保存
save_name = st.sidebar.text_input(
    "保存名", value="my_graph", help="グラフを保存する際のベース名です。"
)
if st.sidebar.button("現在のグラフを保存"):
    if st.session_state.graph:
        saved_folder = save_graph_to_json(st.session_state.graph, save_name)
        if saved_folder:
            st.sidebar.success(f"グラフを `{saved_folder}` に保存しました。")
    else:
        st.sidebar.warning("保存するグラフがありません。先に生成してください。")

# 読込
saved_experiments = get_saved_experiments()
if saved_experiments:
    selected_exp_to_load = st.sidebar.selectbox(
        "ロードするグラフを選択:",
        options=saved_experiments,
        index=None,
        placeholder="保存済みグラフを選択...",
    )
    if st.sidebar.button("選択したグラフをロード"):
        if selected_exp_to_load:
            loaded_graph = load_graph_from_json(selected_exp_to_load)
            if loaded_graph:
                st.session_state.graph = loaded_graph
                st.session_state.graph_name = selected_exp_to_load
                st.sidebar.success(f"`{selected_exp_to_load}` をロードしました。")
                st.rerun()
        else:
            st.sidebar.warning("ロードするグラフを選択してください。")
else:
    st.sidebar.info("保存済みのグラフはありません。")


# --- メイン画面の表示 ---
G = st.session_state.graph

if G:
    st.success(f"**表示中のグラフ:** `{st.session_state.graph_name}`")
    st.write(
        f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}, 有向グラフ: {G.is_directed()}"
    )

    # streamlit-agraph用のノードとエッジを作成
    nodes_vis = [
        Node(id=str(node_id), label=str(node_id), size=15) for node_id in G.nodes()
    ]

    edges_vis = []
    for u, v, data in G.edges(data=True):
        # 'weight' 属性があればラベルとして表示
        edge_label = f"{data.get('weight', 0):.2f}"
        edges_vis.append(Edge(source=str(u), target=str(v), label=edge_label))

    # agraphの描画設定
    config = Config(
        width="100%",
        height=600,
        directed=G.is_directed(),
        physics=False,  # レイアウトを固定する
        hierarchical=False,
        edges={"font": {"size": 12, "align": "top", "color": "#333333"}},
    )

    if nodes_vis:
        agraph(nodes=nodes_vis, edges=edges_vis, config=config)
    else:
        st.write("現在のグラフには表示するノードがありません。")
else:
    st.warning(
        "表示するグラフがありません。サイドバーから新しいグラフを生成するか、保存済みのグラフをロードしてください。"
    )
