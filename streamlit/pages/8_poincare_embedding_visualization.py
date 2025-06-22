import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import os
import json
import sys
from pathlib import Path
import torch
import pandas as pd

# --- パス設定とモジュールのインポート ---
try:
    # スクリプトの実行ディレクトリに基づいてプロジェクトルートを動的に設定
    current_file_dir = Path(__file__).parent
    project_root = current_file_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # HIMモデルのインポート
    from him_full.him_model import HIMModel
    # PyTorchのデバイス設定をインポート
    from him_full.hyperbolic_utils import device
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"必要なモジュールの読み込みに失敗しました: {e}")
    st.info(
        "このアプリを実行するには、プロジェクトのルートディレクトリに `him_full/him_model.py` と `him_full/hyperbolic_utils.py` が必要です。"
    )
    st.stop()


# --- 定数とディレクトリ設定 ---
SAVE_DIR_NAME = "saved_graphs"
SAVE_DIR_PATH = current_file_dir / SAVE_DIR_NAME
if not SAVE_DIR_PATH.exists():
    SAVE_DIR_PATH.mkdir(parents=True, exist_ok=True)


# --- ヘルパー関数 ---
def get_saved_graph_folders():
    """保存ディレクトリからグラフのフォルダリストを取得します。"""
    if not SAVE_DIR_PATH.exists():
        return []
    return sorted(
        [d.name for d in SAVE_DIR_PATH.iterdir() if d.is_dir()],
        reverse=True,
    )

def load_graph_from_json(folder_name):
    """フォルダ名を受け取り、その中のgraph_data.jsonを読み込み、networkxグラフを返します。"""
    filepath = SAVE_DIR_PATH / folder_name / "graph_data.json"
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        G = nx.node_link_graph(data)
        # ノードIDを整数に変換して、後で埋め込みベクトルをスライスしやすくする
        G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes()})
        return G

    except Exception as e:
        st.error(f"グラフの読み込み中にエラーが発生しました: {e}")
        return None


def visualize_2d_embedding(embedding, G, influence_df):
    """Plotlyを使用して埋め込み結果を2次元で可視化する"""
    if embedding is None or G is None:
        return None

    # 可視化には常に最初の2次元を使用
    vis_embedding = embedding[:, :2]

    # マーカーの色を影響度（LDO）に基づいて決定
    influence_map = influence_df.set_index('Node')['Influence (LDO)'].to_dict()
    colors = [influence_map.get(n, 0) for n in G.nodes()]
    
    hover_texts = [
        f'Node {n}<br>Influence (LDO): {influence_map.get(n, 0):.4f}<br>Degree: {G.degree(n)}'
        for n in G.nodes()
    ]

    scatter = go.Scatter(
        x=vis_embedding[:, 0], y=vis_embedding[:, 1], mode='markers',
        marker=dict(
            size=10,
            color=colors,
            colorscale='Reds_r', # 赤色が濃いほど影響度が高い（値が小さい）
            showscale=True,
            colorbar=dict(title='Influence (LDO)'),
            cmin=min(colors) if colors else 0,
            cmax=max(colors) if colors else 1
        ),
        text=hover_texts,
        hoverinfo='text'
    )
    # ポアンカレ円盤の境界線
    circle = go.Scatter(
        x=np.cos(np.linspace(0, 2 * np.pi, 100)),
        y=np.sin(np.linspace(0, 2 * np.pi, 100)),
        mode='lines',
        line=dict(color='black', width=1)
    )
    fig_data = [scatter, circle]
    layout = go.Layout(
        title='2D Poincaré Embedding Visualization',
        xaxis=dict(title='Dimension 1', range=[-1.1, 1.1], zeroline=False),
        yaxis=dict(title='Dimension 2', range=[-1.1, 1.1], scaleanchor="x", scaleratio=1, zeroline=False),
        width=800, height=800, hovermode='closest', showlegend=False
    )

    fig = go.Figure(data=fig_data, layout=layout)
    return fig


# --- セッションステートの初期化 ---
if "hyperbolic_graph" not in st.session_state:
    st.session_state.hyperbolic_graph = None
if "hyperbolic_graph_name" not in st.session_state:
    st.session_state.hyperbolic_graph_name = "未選択"
if "hyperbolic_results" not in st.session_state:
    st.session_state.hyperbolic_results = None

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Hyperbolic Embedding")
st.title("双曲空間埋め込みと影響度分析")

st.markdown("""
このページでは、グラフをHIMモデルで双曲空間へ埋め込み、その結果を**2次元で可視化**します。
スライダーを動かすことで、**学習の進捗（エポック）ごとの埋め込み状態と影響度ランキングの推移**を確認できます。
""")

# --- サイドバー ---
st.sidebar.title("設定")
st.sidebar.header("Step 1: グラフの選択")

saved_folders = get_saved_graph_folders()
if not saved_folders:
    st.sidebar.warning("保存されたグラフがありません。")
else:
    selected_folder = st.sidebar.selectbox(
        "グラフを選択:", [""] + saved_folders, index=0,
        format_func=lambda x: "ファイルを選択" if x == "" else x,
    )
    if st.sidebar.button("グラフを読み込み", disabled=not selected_folder):
        graph = load_graph_from_json(selected_folder)
        if graph:
            st.session_state.hyperbolic_graph = graph
            st.session_state.hyperbolic_graph_name = selected_folder
            st.session_state.hyperbolic_results = None # 新しいグラフを読んだら結果をリセット
            st.toast(f"`{selected_folder}` を読み込みました。", icon="✅")
            st.rerun()

# --- メインコンテンツ ---
G = st.session_state.get("hyperbolic_graph")

if G is None:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"対象グラフ: `{st.session_state.hyperbolic_graph_name}`")
st.metric("ノード数", G.number_of_nodes())
st.markdown("---")

# --- 埋め込み設定 ---
st.sidebar.markdown("---")
st.sidebar.header("Step 2: パラメータ設定")
embedding_dim = st.sidebar.number_input(
    "学習時の埋め込み次元数", min_value=2, max_value=128, value=10, step=2,
    help="学習に用いる双曲空間の次元数です。可視化は常に最初の2次元で行われます。"
)
st.sidebar.subheader("HIMモデル ハイパーパラメータ")
lr = st.sidebar.slider("学習率 (Learning Rate)", 0.001, 0.5, 0.1, 0.001, format="%.3f")
epochs = st.sidebar.slider("エポック数 (Epochs)", 10, 1000, 100, 10)
neg_samples = st.sidebar.number_input("ネガティブサンプル数", 1, 50, 10,
    help="学習時、1つの「つながり有り」ペアに対し、いくつの「つながり無し」ペアを考慮するかを指定します。"
)


if st.sidebar.button("埋め込み生成と影響度分析", type="primary"):
    with st.spinner(f"埋め込み計算と影響度分析を実行中... (デバイス: {device})"):
        try:
            model = HIMModel(
                num_nodes=G.number_of_nodes(), dim=embedding_dim,
                neg_samples=neg_samples, gamma=1.0,
            ).to(device)

            model.fit(G, propagations=[], epochs=epochs, lr=lr, verbose=False)
            
            # 結果をセッションステートに保存
            st.session_state.hyperbolic_results = {
                "embedding_history": model.embeddings_history,
                "ldo_history": model.ldo_history
            }
            st.rerun()

        except Exception as e:
            st.error("処理中に予期せぬエラーが発生しました。")
            st.exception(e)

# --- 結果表示 ---
if st.session_state.get("hyperbolic_results"):
    results = st.session_state.hyperbolic_results
    embedding_history = results["embedding_history"]
    ldo_history = results["ldo_history"]
    
    st.header("分析結果")
    st.success("埋め込みの計算と影響度分析が完了しました。")

    # --- エポック選択スライダー ---
    selected_epoch = st.slider(
        "学習ステップ (Epoch) を選択:",
        min_value=0,
        max_value=len(embedding_history) - 1,
        value=len(embedding_history) - 1, # デフォルトは最終エポック
        help="スライダーを動かして、各学習ステップでの埋め込み状態と影響度を確認できます。"
    )

    # --- 選択されたエポックのデータを処理 ---
    # ローレンツ座標からポアンカレ座標へ変換
    lorentz_emb = embedding_history[selected_epoch]
    gamma = torch.tensor(1.0) # fitで使ったgammaと同じ値
    poincare_emb = lorentz_emb[:, 1:] / (lorentz_emb[:, 0:1] + torch.sqrt(gamma))
    embedding = poincare_emb.numpy()
    
    # 影響度データフレームを作成
    ldo = ldo_history[selected_epoch].numpy()
    nodes = sorted(list(G.nodes()))
    influence_df = pd.DataFrame({
        'Node': nodes,
        'Influence (LDO)': ldo
    }).sort_values('Influence (LDO)', ascending=True).reset_index(drop=True)

    # --- 可視化 ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"2D埋め込み可視化 (Epoch: {selected_epoch + 1})")
        st.caption("色が赤いほど影響度が高い（原点に近い）ことを示します。")
        fig = visualize_2d_embedding(embedding, G, influence_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"影響度ランキング (Epoch: {selected_epoch + 1})")
        st.caption("LDOが小さいほど影響力が高いと解釈されます。")
        st.dataframe(influence_df, use_container_width=True)
