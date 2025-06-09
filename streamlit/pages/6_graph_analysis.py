import streamlit as st
import pandas as pd
import networkx as nx
import json
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語フォントのサポート
import numpy as np
import sys
import os
from collections import Counter

# --- パス設定とモジュールのインポート ---
try:
    # このスクリプト(6_graph_analysis.py)の場所を基準にプロジェクトルートを特定
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"パス設定中にエラーが発生しました: {e}")
    st.stop()

# communityライブラリのインポートを試みる
try:
    import community.community_louvain as community_louvain
except ImportError:
    community_louvain = None  # ライブラリがない場合はNoneに設定

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
if "ga_graph" not in st.session_state:
    st.session_state.ga_graph = None
if "ga_graph_name" not in st.session_state:
    st.session_state.ga_graph_name = "未選択"


# --- サイドバー ---
st.sidebar.title("Graph Analysis")
st.sidebar.header("分析対象グラフの選択")

saved_files = get_saved_graph_files()
if not saved_files:
    st.sidebar.error("読み込み可能なグラフがありません。")
else:
    selected_file = st.sidebar.selectbox(
        "グラフを選択:",
        [""] + saved_files,
        format_func=lambda x: "ファイルを選択" if x == "" else x,
        key="ga_load_selector",
    )
    if st.sidebar.button("グラフを読み込み分析", disabled=not selected_file):
        graph = load_graph_from_json(selected_file)
        if graph:
            st.session_state.ga_graph = graph
            st.session_state.ga_graph_name = selected_file
            st.toast(f"`{selected_file}` を読み込みました。", icon="✅")
            st.rerun()

# --- メインエリア ---
st.title("📊 グラフ構造分析")

G_to_analyze = st.session_state.get("ga_graph")

if G_to_analyze is None:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header(f"分析対象: `{st.session_state.ga_graph_name}`")

# --- 1. 基本統計量 ---
st.subheader("1. 📈 基本統計量")
num_nodes = G_to_analyze.number_of_nodes()
num_edges = G_to_analyze.number_of_edges()
is_directed = G_to_analyze.is_directed()

col1, col2, col3 = st.columns(3)
col1.metric("ノード数", num_nodes)
col2.metric("エッジ数", num_edges)
col3.metric("密度", f"{nx.density(G_to_analyze):.4f}")

if num_nodes > 0:
    if is_directed:
        in_degrees = [d for n, d in G_to_analyze.in_degree()]
        out_degrees = [d for n, d in G_to_analyze.out_degree()]
        st.write(f"- 平均入次数: {np.mean(in_degrees):.2f}")
        st.write(f"- 平均出次数: {np.mean(out_degrees):.2f}")
    else:
        degrees = [d for n, d in G_to_analyze.degree()]
        st.write(f"- 平均次数: {np.mean(degrees):.2f}")

    st.markdown("**次数分布**")
    cols_deg_hist = st.columns(2 if is_directed else 1)
    bin_count = max(1, min(30, int(num_nodes / 10)))

    if is_directed:
        with cols_deg_hist[0]:
            fig_in, ax_in = plt.subplots()
            ax_in.hist(in_degrees, bins=bin_count, color="skyblue", edgecolor="black")
            ax_in.set_title("入次数分布")
            ax_in.set_xlabel("入次数")
            ax_in.set_ylabel("ノード数")
            st.pyplot(fig_in)
        with cols_deg_hist[1]:
            fig_out, ax_out = plt.subplots()
            ax_out.hist(
                out_degrees, bins=bin_count, color="lightcoral", edgecolor="black"
            )
            ax_out.set_title("出次数分布")
            ax_out.set_xlabel("出次数")
            ax_out.set_ylabel("ノード数")
            st.pyplot(fig_out)
    else:  # Undirected
        with cols_deg_hist[0]:
            fig, ax = plt.subplots()
            ax.hist(degrees, bins=bin_count, color="mediumseagreen", edgecolor="black")
            ax.set_title("次数分布")
            ax.set_xlabel("次数")
            ax.set_ylabel("ノード数")
            st.pyplot(fig)

st.markdown("---")

# --- 2. グラフ構造の性質 ---
st.subheader("2. 🔗 グラフ構造の性質")
if num_nodes > 0:
    st.markdown("**連結性**")
    if is_directed:
        with st.spinner("連結性を計算中..."):
            sccs = list(nx.strongly_connected_components(G_to_analyze))
            largest_scc_size = len(max(sccs, key=len)) if sccs else 0
            wccs = list(nx.weakly_connected_components(G_to_analyze))
            largest_wcc_size = len(max(wccs, key=len)) if wccs else 0
        st.write(f"- 強連結成分 (SCC) の数: {len(sccs)}")
        st.write(
            f"- 最大SCCのサイズ: {largest_scc_size} ({largest_scc_size/num_nodes:.1%})"
        )
        st.write(f"- 弱連結成分 (WCC) の数: {len(wccs)}")
        st.write(
            f"- 最大WCCのサイズ: {largest_wcc_size} ({largest_wcc_size/num_nodes:.1%})"
        )
    else:  # Undirected
        ccs = list(nx.connected_components(G_to_analyze))
        largest_cc_size = len(max(ccs, key=len)) if ccs else 0
        st.write(f"- 連結成分の数: {len(ccs)}")
        st.write(
            f"- 最大連結成分のサイズ: {largest_cc_size} ({largest_cc_size/num_nodes:.1%})"
        )

    st.markdown("**クラスタリング係数**")
    with st.spinner("クラスタリング係数を計算中..."):
        avg_clustering = nx.average_clustering(G_to_analyze)
        transitivity = nx.transitivity(G_to_analyze)
    st.write(f"- 平均クラスタリング係数: {avg_clustering:.4f}")
    st.write(f"- 推移性 (グローバル): {transitivity:.4f}")

    st.markdown("**平均最短経路長**")
    if num_nodes > 1:
        with st.spinner("平均最短経路長を計算中... (時間がかかる場合があります)"):
            try:
                components = list(
                    nx.weakly_connected_components(G_to_analyze)
                    if is_directed
                    else nx.connected_components(G_to_analyze)
                )
                if components:
                    largest_comp_nodes = max(components, key=len)
                    if len(largest_comp_nodes) > 1:
                        largest_comp_graph = G_to_analyze.subgraph(largest_comp_nodes)
                        avg_path = nx.average_shortest_path_length(largest_comp_graph)
                        st.write(f"- 平均最短経路長 (最大連結成分内): {avg_path:.2f}")
                    else:
                        st.write("- 計算可能な連結成分がありません。")
            except Exception as e:
                st.warning(f"- 計算中にエラーが発生しました: {e}")
    else:
        st.write("- ノードが1つ以下のため計算できません。")

    if community_louvain:
        st.markdown("**コミュニティ構造 (Louvain法)**")
        with st.spinner("コミュニティを検出中..."):
            try:
                G_undirected = G_to_analyze.to_undirected()
                partition = community_louvain.best_partition(G_undirected)
                modularity = community_louvain.modularity(partition, G_undirected)
                st.write(f"- 検出されたコミュニティ数: {len(set(partition.values()))}")
                st.write(f"- モジュラリティ: {modularity:.4f}")
            except Exception as e:
                st.warning(f"- コミュニティ検出中にエラー: {e}")
    else:
        st.markdown("**コミュニティ構造 (Louvain法)**")
        st.info(
            "`python-louvain`がインストールされていないため、スキップします。(`pip install python-louvain`でインストールできます)"
        )


st.markdown("---")

# --- 3. 中心性指標 ---
st.subheader("3. 🌟 中心性指標 (上位5ノード)")
if num_nodes > 0:
    with st.spinner("各種中心性を計算中... (媒介中心性などは時間がかかります)"):
        k_top = 5
        centrality_calculators = {}
        if is_directed:
            centrality_calculators["入次数中心性"] = nx.in_degree_centrality
            centrality_calculators["出次数中心性"] = nx.out_degree_centrality
        else:
            centrality_calculators["次数中心性"] = nx.degree_centrality

        centrality_calculators["PageRank"] = nx.pagerank
        centrality_calculators["媒介中心性"] = nx.betweenness_centrality
        centrality_calculators["近接中心性"] = nx.closeness_centrality

        results_data = []
        for name, func in centrality_calculators.items():
            try:
                centrality_dict = func(G_to_analyze)
                top_nodes = sorted(
                    centrality_dict.items(), key=lambda x: x[1], reverse=True
                )[:k_top]
                row = {"Centrality": name}
                for i, (node, score) in enumerate(top_nodes, 1):
                    row[f"Top {i} Node"] = node
                    row[f"Top {i} Score"] = f"{score:.4f}"
                results_data.append(row)
            except Exception as e:
                st.warning(f"{name}の計算中にエラー: {e}")

    if results_data:
        results_df = pd.DataFrame(results_data).set_index("Centrality")
        st.dataframe(results_df)
else:
    st.write("ノードがないため、中心性を計算できません。")

st.markdown("---")

# --- 4. その他の性質 ---
st.subheader("4. 🧩 その他の性質")
if num_nodes > 1:
    st.markdown("**次数相関 (Degree Assortativity)**")
    with st.spinner("次数相関を計算中..."):
        try:
            assortativity = nx.degree_assortativity_coefficient(G_to_analyze)
            st.write(f"- 次数相関 (ピアソン相関係数): {assortativity:.4f}")
            if assortativity > 0.1:
                st.info(
                    "💡 **同類選択的 (Assortative):** 次数の高いノード同士が接続しやすい傾向があります。"
                )
            elif assortativity < -0.1:
                st.info(
                    "💡 **異類選択的 (Disassortative):** 次数の高いノードと低いノードが接続しやすい傾向があります。"
                )
            else:
                st.info("💡 **中立 (Neutral):** 次数に関する特定の相関は見られません。")
        except Exception as e:
            st.warning(f"- 次数相関の計算中にエラー: {e}")

    st.markdown("**リッチクラブ係数 (Rich-club Coefficient)**")
    if num_edges > 0:
        with st.spinner("リッチクラブ係数を計算中..."):
            try:
                G_undirected = G_to_analyze.to_undirected()
                rc = nx.rich_club_coefficient(G_undirected, normalized=False)
                if rc:
                    fig_rc, ax_rc = plt.subplots()
                    ax_rc.plot(
                        list(rc.keys()), list(rc.values()), marker="o", linestyle="-"
                    )
                    ax_rc.set_xlabel("次数 k")
                    ax_rc.set_ylabel("リッチクラブ係数 φ(k)")
                    ax_rc.set_title(
                        "リッチクラブ係数（次数k以上のノード間の密な結合度）"
                    )
                    ax_rc.grid(True)
                    st.pyplot(fig_rc)
                else:
                    st.write("- リッチクラブ係数: 計算結果が空でした。")
            except Exception as e:
                st.warning(f"- リッチクラブ係数の計算中にエラー: {e}")
    else:
        st.write("- リッチクラブ係数: エッジがないため計算できません。")

    if is_directed:
        st.markdown("**トライアド構造 (Triadic Census)**")
        with st.spinner("トライアド構造を計算中... (時間がかかる場合があります)"):
            try:
                if num_nodes >= 3:
                    census = nx.triadic_census(G_to_analyze)
                    df_census = pd.DataFrame(
                        list(census.items()), columns=["Motif ID", "Count"]
                    )
                    df_census = (
                        df_census[df_census["Count"] > 0]
                        .sort_values(by="Count", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.write("3ノード間の関係性の種類と数:")
                    st.dataframe(df_census)
                    st.caption(
                        "Motif IDは3ノード間の関係パターンを示します (例: 030Tは3サイクル、102は相互リンク)。"
                    )
                else:
                    st.write("- ノード数が3未満のため計算できません。")
            except Exception as e:
                st.warning(f"- トライアド分析の計算中にエラー: {e}")

st.markdown("---")
st.caption(
    "注: 一部の計算、特に媒介中心性や最短経路長などは、グラフのサイズや構造によって非常に時間がかかる場合があります。"
)
