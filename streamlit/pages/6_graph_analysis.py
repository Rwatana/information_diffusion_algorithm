import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語フォントのサポート
import numpy as np
import sys
import os
from collections import Counter # Counterをインポート

# --- パス設定 ---
current_dir = os.path.dirname(__file__) # この行はStreamlit環境では __file__ が期待通りに動作しないことがあるため、ローカル実行時のみ有効
streamlit_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root_dir = os.path.abspath(os.path.join(streamlit_dir, '..'))
sys.path.append(project_root_dir)

from datagen.data_utils import generate_graph # 必要に応じて

st.set_page_config(layout="wide", page_title="グラフ構造分析")
st.title("📊 グラフ構造分析") # タイトルに絵文字を追加

# --- サイドバーでのグラフ選択/アップロード機能 (将来的に2つ入力できるように拡張) ---
st.sidebar.header("分析対象グラフ")

# グラフ可視化タブで生成されたグラフを使用するオプション
use_generated_graph = st.sidebar.checkbox("「グラフ可視化」タブのグラフを使用する", value=True, key="ga_use_generated")

graph_sources = {} # 分析対象グラフを格納する辞書 {name: G_object}

if use_generated_graph:
    graph_key = 'gv_graph' # グラフ可視化ページで設定されたグラフのキー
    if graph_key in st.session_state and st.session_state[graph_key] is not None:
        G_main = st.session_state[graph_key]
        # グラフが空でないか、またはNoneでないかを確認
        if G_main is not None and G_main.number_of_nodes() > 0 :
            graph_sources["生成されたグラフ"] = G_main
            st.sidebar.success("「グラフ可視化」タブのグラフをロードしました。")
        elif G_main is not None and G_main.number_of_nodes() == 0:
            st.sidebar.warning("「グラフ可視化」タブのグラフはノード数が0です。")
            G_main = None # 分析対象外とする
        else:
            st.sidebar.warning("「グラフ可視化」タブでグラフが適切に生成されていません。")
            G_main = None
    else:
        st.sidebar.warning("「グラフ可視化」タブでグラフが生成されていません。")
        G_main = None
else:
    # 将来的にはファイルアップロード機能などをここに追加
    st.sidebar.info("現在は「グラフ可視化」タブのグラフのみ利用可能です。")
    G_main = None


# --- 分析の実行と表示 ---
if not graph_sources:
    st.info("サイドバーで分析対象のグラフをロードまたは生成してください。")
    st.stop()

selected_graph_name = list(graph_sources.keys())[0]
G_to_analyze = graph_sources[selected_graph_name]

st.header(f"分析対象: {selected_graph_name}")

if G_to_analyze is None or G_to_analyze.number_of_nodes() == 0:
    st.warning(f"分析対象のグラフ「{selected_graph_name}」にノードがないか、ロードされていません。")
    st.stop()

# --- 1. 基本統計量 ---
st.subheader("1. 📈 基本統計量")
num_nodes = G_to_analyze.number_of_nodes()
num_edges = G_to_analyze.number_of_edges()
st.write(f"- ノード数 (ユーザー数): {num_nodes}")
st.write(f"- エッジ数 (フォロー関係数): {num_edges}")

if num_nodes > 0:
    # 平均次数
    if G_to_analyze.is_directed():
        in_degrees_list = [d for n, d in G_to_analyze.in_degree()]
        out_degrees_list = [d for n, d in G_to_analyze.out_degree()]
        avg_in_degree = sum(in_degrees_list) / num_nodes if num_nodes > 0 else 0
        avg_out_degree = sum(out_degrees_list) / num_nodes if num_nodes > 0 else 0
        st.write(f"- 平均入次数 (平均フォロワー数): {avg_in_degree:.2f}")
        st.write(f"- 平均出次数 (平均フォロー数): {avg_out_degree:.2f}")
        degrees_list = in_degrees_list + out_degrees_list # 総合的な次数として
    else:
        degrees_list = [d for n, d in G_to_analyze.degree()]
        avg_degree = sum(degrees_list) / num_nodes if num_nodes > 0 else 0
        st.write(f"- 平均次数: {avg_degree:.2f}")

    # 密度
    density = nx.density(G_to_analyze)
    max_possible_edges_formula = "N*(N-1)" if G_to_analyze.is_directed() else "N*(N-1)/2"
    max_possible_edges = num_nodes * (num_nodes - 1) if G_to_analyze.is_directed() else num_nodes * (num_nodes - 1) / 2
    max_possible_edges = max(1, max_possible_edges) # 0除算を避ける
    st.write(f"- 密度: {density:.4f} (最大可能エッジ数 {max_possible_edges_formula}: {int(max_possible_edges)})")

    # 次数分布
    st.markdown("**次数分布 (Degree Distribution)**")
    
    # 次数の統計的記述
    if degrees_list: # degrees_listが空でないことを確認
        st.write(f"  - 次数の中央値: {np.median(degrees_list):.2f}, 最大次数: {np.max(degrees_list)}, 最小次数: {np.min(degrees_list)}")

    cols_deg_hist = st.columns(2 if G_to_analyze.is_directed() else 1)
    bin_count = max(1, min(30, int(num_nodes/10) if num_nodes > 10 else num_nodes)) # ビン数を調整

    if G_to_analyze.is_directed():
        with cols_deg_hist[0]:
            st.write("入次数分布 (フォロワー数)")
            fig_in_deg, ax_in_deg = plt.subplots()
            ax_in_deg.hist(in_degrees_list, bins=bin_count, rwidth=0.9, color='skyblue', edgecolor='black')
            ax_in_deg.set_xlabel("入次数")
            ax_in_deg.set_ylabel("ノード数")
            ax_in_deg.grid(axis='y', alpha=0.75)
            st.pyplot(fig_in_deg)

        with cols_deg_hist[1]:
            st.write("出次数分布 (フォロー数)")
            fig_out_deg, ax_out_deg = plt.subplots()
            ax_out_deg.hist(out_degrees_list, bins=bin_count, rwidth=0.9, color='lightcoral', edgecolor='black')
            ax_out_deg.set_xlabel("出次数")
            ax_out_deg.set_ylabel("ノード数")
            ax_out_deg.grid(axis='y', alpha=0.75)
            st.pyplot(fig_out_deg)
        
        # 両対数プロット (次数分布がべき乗則に従うか確認)
        st.markdown("**次数分布の両対数プロット (Log-Log Scale)**")
        cols_loglog = st.columns(2)
        with cols_loglog[0]:
            in_degree_counts = Counter(in_degrees_list)
            if in_degree_counts: # Counterが空でないことを確認
                in_deg, in_cnt = zip(*sorted(in_degree_counts.items()))
                fig_in_log, ax_in_log = plt.subplots()
                ax_in_log.loglog(in_deg, in_cnt, marker='o', linestyle='none', color='skyblue')
                ax_in_log.set_xlabel("入次数 (Log Scale)")
                ax_in_log.set_ylabel("ノード数 (Log Scale)")
                ax_in_log.set_title("入次数 (Log-Log)")
                ax_in_log.grid(True, which="both", ls="-", alpha=0.5)
                st.pyplot(fig_in_log)
            else:
                st.write("入次数データがありません。")
        with cols_loglog[1]:
            out_degree_counts = Counter(out_degrees_list)
            if out_degree_counts: # Counterが空でないことを確認
                out_deg, out_cnt = zip(*sorted(out_degree_counts.items()))
                fig_out_log, ax_out_log = plt.subplots()
                ax_out_log.loglog(out_deg, out_cnt, marker='o', linestyle='none', color='lightcoral')
                ax_out_log.set_xlabel("出次数 (Log Scale)")
                ax_out_log.set_ylabel("ノード数 (Log Scale)")
                ax_out_log.set_title("出次数 (Log-Log)")
                ax_out_log.grid(True, which="both", ls="-", alpha=0.5)
                st.pyplot(fig_out_log)
            else:
                st.write("出次数データがありません。")


    else: # 無向グラフの場合
        with cols_deg_hist[0]:
            st.write("次数分布")
            fig_deg, ax_deg = plt.subplots()
            ax_deg.hist(degrees_list, bins=bin_count, rwidth=0.9, color='mediumseagreen', edgecolor='black')
            ax_deg.set_xlabel("次数")
            ax_deg.set_ylabel("ノード数")
            ax_deg.grid(axis='y', alpha=0.75)
            st.pyplot(fig_deg)

        st.markdown("**次数分布の両対数プロット (Log-Log Scale)**")
        degree_counts = Counter(degrees_list)
        if degree_counts: # Counterが空でないことを確認
            deg, cnt = zip(*sorted(degree_counts.items()))
            fig_log, ax_log = plt.subplots()
            ax_log.loglog(deg, cnt, marker='o', linestyle='none', color='mediumseagreen')
            ax_log.set_xlabel("次数 (Log Scale)")
            ax_log.set_ylabel("ノード数 (Log Scale)")
            ax_log.set_title("次数 (Log-Log)")
            ax_log.grid(True, which="both", ls="-", alpha=0.5)
            st.pyplot(fig_log)
        else:
            st.write("次数データがありません。")


st.markdown("---")
# --- 2. グラフ構造の性質 ---
st.subheader("2. 🔗 グラフ構造の性質")
# 連結性
st.markdown("**連結性 (Connectivity)**")
if G_to_analyze.is_directed():
    num_sccs = nx.number_strongly_connected_components(G_to_analyze)
    sccs = list(nx.strongly_connected_components(G_to_analyze))
    largest_scc_size = len(max(sccs, key=len)) if sccs else 0
    st.write(f"- 強連結成分 (SCCs) の数: {num_sccs}")
    st.write(f"- 最大強連結成分のサイズ: {largest_scc_size} ({largest_scc_size/num_nodes*100 if num_nodes > 0 else 0:.1f}%)")
    if num_sccs == 1 and largest_scc_size == num_nodes:
        st.success("  - ✅ このグラフは強連結です（既約）。")
    if sccs:
        scc_sizes = [len(s) for s in sccs]
        fig_scc, ax_scc = plt.subplots()
        ax_scc.hist(scc_sizes, bins=max(1, min(10, len(scc_sizes))), rwidth=0.9, color='cyan', edgecolor='black')
        ax_scc.set_title("強連結成分のサイズ分布")
        ax_scc.set_xlabel("SCCサイズ")
        ax_scc.set_ylabel("SCC数")
        st.pyplot(fig_scc)


    num_wccs = nx.number_weakly_connected_components(G_to_analyze)
    wccs = list(nx.weakly_connected_components(G_to_analyze))
    largest_wcc_size = len(max(wccs, key=len)) if wccs else 0
    st.write(f"- 弱連結成分 (WCCs) の数: {num_wccs}")
    st.write(f"- 最大弱連結成分のサイズ: {largest_wcc_size} ({largest_wcc_size/num_nodes*100 if num_nodes > 0 else 0:.1f}%)")
    if num_wccs == 1 and largest_wcc_size == num_nodes:
        st.success("  - ✅ このグラフは弱連結です。")
    if wccs:
        wcc_sizes = [len(s) for s in wccs]
        fig_wcc, ax_wcc = plt.subplots()
        ax_wcc.hist(wcc_sizes, bins=max(1, min(10, len(wcc_sizes))), rwidth=0.9, color='magenta', edgecolor='black')
        ax_wcc.set_title("弱連結成分のサイズ分布")
        ax_wcc.set_xlabel("WCCサイズ")
        ax_wcc.set_ylabel("WCC数")
        st.pyplot(fig_wcc)


else: # 無向グラフ
    num_ccs = nx.number_connected_components(G_to_analyze)
    ccs = list(nx.connected_components(G_to_analyze))
    largest_cc_size = len(max(ccs, key=len)) if ccs else 0
    st.write(f"- 連結成分の数: {num_ccs}")
    st.write(f"- 最大連結成分のサイズ: {largest_cc_size} ({largest_cc_size/num_nodes*100 if num_nodes > 0 else 0:.1f}%)")
    if num_ccs == 1 and largest_cc_size == num_nodes:
        st.success("  - ✅ このグラフは連結です。")
    if ccs:
        cc_sizes = [len(s) for s in ccs]
        fig_cc, ax_cc = plt.subplots()
        ax_cc.hist(cc_sizes, bins=max(1, min(10, len(cc_sizes))), rwidth=0.9, color='orange', edgecolor='black')
        ax_cc.set_title("連結成分のサイズ分布")
        ax_cc.set_xlabel("CCサイズ")
        ax_cc.set_ylabel("CC数")
        st.pyplot(fig_cc)

    # ブリッジと関節点の検出 (無向グラフのみ)
    if num_nodes < 500: # 計算量が多いので制限
        try:
            if num_nodes > 0: # ノードがないとエラーになるため
                bridges = list(nx.bridges(G_to_analyze))
                articulation_points = list(nx.articulation_points(G_to_analyze))
                st.write(f"- ブリッジ (除去で連結成分が増えるエッジ) の数: {len(bridges)}")
                st.write(f"- 関節点 (除去で連結成分が増えるノード) の数: {len(articulation_points)}")
            else:
                st.write("- ブリッジ/関節点: ノード数が0のため計算できません。")
        except Exception as e:
            st.write(f"- ブリッジ/関節点の計算中にエラー: {e}")
    else:
        st.write("- ブリッジ/関節点: グラフサイズが大きいため計算をスキップ。")


# クラスタリング係数
st.markdown("**クラスタリング係数 (Clustering Coefficient)**")
avg_clustering = nx.average_clustering(G_to_analyze)
st.write(f"- 平均ローカルクラスタリング係数: {avg_clustering:.4f}")

# グローバルクラスタリング係数 (推移性)
transitivity = nx.transitivity(G_to_analyze)
st.write(f"- グローバルクラスタリング係数 (推移性): {transitivity:.4f}")

# ローカルクラスタリング係数の分布
if num_nodes > 0:
    local_clustering_coeffs = list(nx.clustering(G_to_analyze).values())
    if local_clustering_coeffs: # リストが空でないことを確認
        fig_lcc, ax_lcc = plt.subplots()
        ax_lcc.hist(local_clustering_coeffs, bins=20, rwidth=0.9, color='gold', edgecolor='black')
        ax_lcc.set_title("ローカルクラスタリング係数の分布")
        ax_lcc.set_xlabel("ローカルクラスタリング係数")
        ax_lcc.set_ylabel("ノード数")
        ax_lcc.grid(axis='y', alpha=0.75)
        st.pyplot(fig_lcc)
    else:
        st.write("ローカルクラスタリング係数データがありません。")


# 平均最短経路長と直径
st.markdown("**経路長と直径 (Path Length & Diameter)**")
path_length_calculated = False
if num_nodes < 300 and num_nodes > 1: # ノード数が少ない(かつ2以上)場合のみ計算
    try:
        if G_to_analyze.is_directed():
            if sccs and largest_scc_size > 1 :
                largest_scc_graph = G_to_analyze.subgraph(max(sccs, key=len)).copy()
                if nx.is_strongly_connected(largest_scc_graph): # 強連結であることを確認
                     avg_shortest_path = nx.average_shortest_path_length(largest_scc_graph)
                     st.write(f"- 平均最短経路長 (最大SCC内, {largest_scc_graph.number_of_nodes()}ノード): {avg_shortest_path:.2f}")
                     path_length_calculated = True
                     # diameter = nx.diameter(largest_scc_graph) # 直径も同様
                     # st.write(f"- 直径 (最大SCC内): {diameter}")
            # 最大WCCがグラフ全体に近く、ノード数が複数あれば、有向のままで計算を試みる
            elif wccs and largest_wcc_size > 1: # and largest_wcc_size > num_nodes * 0.5:
                largest_wcc_subgraph = G_to_analyze.subgraph(max(wccs, key=len))
                try:
                    avg_shortest_path_wcc_dir = nx.average_shortest_path_length(largest_wcc_subgraph)
                    st.write(f"- 平均最短経路長 (最大WCC内, 有向, {largest_wcc_subgraph.number_of_nodes()}ノード): {avg_shortest_path_wcc_dir:.2f}")
                    path_length_calculated = True
                except nx.NetworkXError:
                     st.write(f"- 平均最短経路長 (最大WCC内, {largest_wcc_subgraph.number_of_nodes()}ノード): 弱連結成分が強連結でないため、一部のノードペア間で到達不可能です。")

        else: # 無向グラフ
            if ccs and largest_cc_size > 1:
                largest_cc_graph = G_to_analyze.subgraph(max(ccs, key=len)).copy()
                if nx.is_connected(largest_cc_graph):
                    avg_shortest_path = nx.average_shortest_path_length(largest_cc_graph)
                    st.write(f"- 平均最短経路長 (最大連結成分内, {largest_cc_graph.number_of_nodes()}ノード): {avg_shortest_path:.2f}")
                    path_length_calculated = True
                    # diameter = nx.diameter(largest_cc_graph)
                    # st.write(f"- 直径 (最大連結成分内): {diameter}")

        if not path_length_calculated:
            st.write("- 平均最短経路長: 計算対象の適切な(大きな)連結/強連結成分が見つからないか、計算できませんでした。")

    except Exception as e:
        st.warning(f"- 平均最短経路長/直径の計算中にエラー: {e}")
elif num_nodes <=1:
    st.write("- 平均最短経路長/直径: ノード数が1以下のため計算できません。")
else: # num_nodes >= 300
    st.write("- 平均最短経路長/直径: グラフサイズが大きいため、主要な連結成分での計算をスキップしました。")

# コミュニティ構造 (Louvain法)
st.markdown("**コミュニティ構造 (Community Structure)**")
try:
    import community as community_louvain # python-louvain
    G_community_analysis = G_to_analyze
    
    if not G_to_analyze.is_directed() and G_community_analysis.number_of_nodes() > 0:
        partition = community_louvain.best_partition(G_community_analysis)
        num_communities = len(set(partition.values()))
        st.write(f"- Louvain法によるコミュニティ検出 (無向グラフ): {num_communities}個のコミュニティ")
        if num_communities > 0 and num_communities < G_community_analysis.number_of_nodes(): # 全ノードが別コミュニティは除く
            modularity = community_louvain.modularity(partition, G_community_analysis)
            st.write(f"  - モジュラリティ: {modularity:.4f}")
            community_sizes_counter = Counter(partition.values())
            community_sizes_list = list(community_sizes_counter.values())
            
            fig_com, ax_com = plt.subplots()
            ax_com.hist(community_sizes_list, bins=max(1, min(20, num_communities)), rwidth=0.9, color='purple', edgecolor='black')
            ax_com.set_title("コミュニティサイズ分布 (Louvain)")
            ax_com.set_xlabel("コミュニティサイズ")
            ax_com.set_ylabel("コミュニティ数")
            st.pyplot(fig_com)
        elif num_communities == G_community_analysis.number_of_nodes():
            st.write("  - 各ノードが個別のコミュニティとして検出されました。")
        else:
            st.write("  - コミュニティは検出されませんでした。")
    elif G_to_analyze.is_directed():
         st.write("- Louvain法: 有向グラフには直接適用できません。無向グラフに変換 (`G.to_undirected()`) してお試しください。")
    else: # ノード数0の場合
        st.write("- コミュニティ構造 (Louvain法): ノード数が0のためスキップ。")

except ImportError:
    st.write("- コミュニティ構造 (Louvain法): `python-louvain`ライブラリが見つかりません (`pip install python-louvain`)。")
except Exception as e:
    st.write(f"- コミュニティ構造の計算中にエラー: {e}")

st.markdown("---")
# --- 3. 中心性指標 ---
st.subheader("3. 🌟 中心性指標 (Centrality Measures)")
st.write("上位5ノードを表示します。")
k_top_centrality = 5

# 次数中心性
st.markdown("**次数中心性 (Degree Centrality)**")
if G_to_analyze.is_directed():
    in_degree_centrality = nx.in_degree_centrality(G_to_analyze)
    top_in_degree = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
    st.write(f"- 入次数中心性: {[(n, f'{s:.3f}') for n, s in top_in_degree]}")

    out_degree_centrality = nx.out_degree_centrality(G_to_analyze)
    top_out_degree = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
    st.write(f"- 出次数中心性: {[(n, f'{s:.3f}') for n, s in top_out_degree]}")
else:
    degree_centrality = nx.degree_centrality(G_to_analyze)
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
    st.write(f"- 次数中心性: {[(n, f'{s:.3f}') for n, s in top_degree]}")

# PageRank
st.markdown("**PageRank**")
if num_nodes > 0:
    try:
        pagerank = nx.pagerank(G_to_analyze, alpha=0.85, max_iter=100, tol=1.0e-6)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        st.write(f"- PageRank (alpha=0.85): {[(n, f'{s:.4f}') for n, s in top_pagerank]}")
    except nx.PowerIterationFailedConvergence:
        st.warning("- PageRank: 計算が収束しませんでした。試行回数を増やすか、許容誤差を調整してください。")
    except Exception as e:
        st.error(f"- PageRankの計算中にエラー: {e}")
else:
    st.write("- PageRank: ノード数が0のため計算できません。")


# HITS (Hubs and Authorities) - 有向グラフのみ
if G_to_analyze.is_directed() and num_nodes > 0:
    st.markdown("**HITS (Hubs and Authorities)**")
    try:
        hubs, authorities = nx.hits(G_to_analyze, max_iter=100, tol=1.0e-6)
        top_hubs = sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        top_authorities = sorted(authorities.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        st.write(f"- Hubスコア上位: {[(n, f'{s:.4f}') for n, s in top_hubs]}")
        st.write(f"- Authorityスコア上位: {[(n, f'{s:.4f}') for n, s in top_authorities]}")
    except nx.PowerIterationFailedConvergence:
        st.warning("- HITS: 計算が収束しませんでした。")
    except Exception as e:
        st.error(f"- HITSの計算中にエラー: {e}")
elif not G_to_analyze.is_directed():
    st.markdown("**HITS (Hubs and Authorities)**")
    st.write("- HITS: 無向グラフのため適用されません (有向グラフの指標です)。")


# 固有ベクトル中心性
st.markdown("**固有ベクトル中心性 (Eigenvector Centrality)**")
if num_nodes > 0:
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_to_analyze, max_iter=100, tol=1.0e-6)
        top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        st.write(f"- 固有ベクトル中心性: {[(n, f'{s:.4f}') for n, s in top_eigenvector]}")
    except nx.PowerIterationFailedConvergence:
         st.warning("- 固有ベクトル中心性: 計算が収束しませんでした。")
    except Exception as e: # Catch more general errors too, e.g. for disconnected graphs if not handled by nx
        st.error(f"- 固有ベクトル中心性の計算中にエラー: {e}")
else:
    st.write("- 固有ベクトル中心性: ノード数が0のため計算できません。")


# 媒介中心性 (計算量が多いので注意)
st.markdown("**媒介中心性 (Betweenness Centrality)**")
if num_nodes < 200 and num_nodes > 2: # 小さなグラフでのみ全計算 (3ノード以上)
    try:
        betweenness_centrality = nx.betweenness_centrality(G_to_analyze, normalized=True, endpoints=False)
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        st.write(f"- 媒介中心性 (全ノード): {[(n, f'{s:.4f}') for n, s in top_betweenness]}")
    except Exception as e:
        st.error(f"- 媒介中心性の計算中にエラー: {e}")
elif num_nodes >= 200 and num_nodes < 1000: # 中規模グラフではサンプル計算
    try:
        sample_k = min(max(10, int(num_nodes * 0.1)), 100)
        betweenness_centrality_sampled = nx.betweenness_centrality(G_to_analyze, k=sample_k, normalized=True, endpoints=False)
        top_betweenness_sampled = sorted(betweenness_centrality_sampled.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        st.write(f"- 媒介中心性 (サンプル計算 k={sample_k}): {[(n, f'{s:.4f}') for n, s in top_betweenness_sampled]}")
    except Exception as e:
        st.error(f"- 媒介中心性 (サンプル計算) のエラー: {e}")
elif num_nodes <= 2 and num_nodes > 0:
    st.write("- 媒介中心性: ノード数が少なすぎるため計算できません (3ノード以上必要)。")
else: # num_nodes >= 1000 or num_nodes == 0
    st.write("- 媒介中心性: グラフサイズが大きいかノードがないため、計算をスキップしました。")


# 近接中心性
st.markdown("**近接中心性 (Closeness Centrality)**")
if num_nodes < 500 and num_nodes > 0:
    try:
        closeness_centrality = nx.closeness_centrality(G_to_analyze) # wf_improved is deprecated
        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:k_top_centrality]
        st.write(f"- 近接中心性: {[(n, f'{s:.4f}') for n, s in top_closeness]}")
    except Exception as e:
        st.error(f"- 近接中心性の計算中にエラー: {e}")
        st.caption("  - グラフが非常に小さい、または特殊な構造(例:非連結で一部のノードが孤立)の場合に発生することがあります。")
else:
    st.write("- 近接中心性: グラフサイズが大きいかノードがないため計算をスキップしました。")

st.markdown("---")
# --- 4. その他の性質 ---
st.subheader("4. 🧩 その他の性質")

# 次数相関 (Assortativity)
st.markdown("**次数相関 (Degree Assortativity)**")
if num_edges > 0 and num_nodes > 1: # エッジがあり、複数のノードが存在する場合
    try:
        assortativity = nx.degree_assortativity_coefficient(G_to_analyze)
        st.write(f"- 次数相関 (ピアソン相関係数): {assortativity:.4f}")
        if assortativity > 0.1:
            st.write("  - 傾向: 次数の高いノード同士が接続しやすい (Assortative mixing)")
        elif assortativity < -0.1:
            st.write("  - 傾向: 次数の高いノードと低いノードが接続しやすい (Disassortative mixing)")
        else:
            st.write("  - 傾向: 特定の相関は見られない (Neutral)")
    except Exception as e: # 例えば、全てのノードの次数が同じ場合など
        st.warning(f"- 次数相関の計算中にエラー: {e} (グラフの次数分布が均一すぎる場合などに発生することがあります)")
else:
    st.write("- 次数相関: エッジがないか、ノード数が1以下なため計算できません。")


# リッチクラブ係数 (Rich-club coefficient)
st.markdown("**リッチクラブ係数 (Rich-club Coefficient)**")
if num_nodes > 10 and num_edges > 0 :
    try:
        G_for_richclub = G_to_analyze.to_undirected() if G_to_analyze.is_directed() else G_to_analyze
        
        if G_for_richclub.number_of_edges() > 0:
            # Check if there are varying degrees to compute rich-club
            degrees_rc = [d for n,d in G_for_richclub.degree()]
            if len(set(degrees_rc)) > 1 : # 複数の異なる次数が存在する場合
                rc_all_k = nx.rich_club_coefficient(G_for_richclub, normalized=False)
                if rc_all_k:
                    fig_rc, ax_rc = plt.subplots()
                    ax_rc.plot(list(rc_all_k.keys()), list(rc_all_k.values()), marker='o', linestyle='-')
                    ax_rc.set_xlabel("次数 k")
                    ax_rc.set_ylabel("リッチクラブ係数 φ(k)")
                    ax_rc.set_title("リッチクラブ係数 (正規化なし)")
                    ax_rc.grid(True)
                    st.pyplot(fig_rc)
                else:
                    st.write("- リッチクラブ係数: 計算結果が空でした（適切な次数kの範囲がなかった可能性があります）。")
            else:
                st.write("- リッチクラブ係数: 全てのノードの次数が同じため、有益な計算ができません。")
        else:
             st.write("- リッチクラブ係数: (無向変換後の)グラフにエッジがないため計算できません。")
    except Exception as e:
        st.warning(f"- リッチクラブ係数の計算中にエラー: {e}")
else:
    st.write("- リッチクラブ係数: グラフサイズが小さいかエッジがないため計算をスキップ。")

# トライアド構造 (Triadic Census) - 有向グラフのみ
if G_to_analyze.is_directed() and num_nodes >= 3 and num_nodes < 300:
    st.markdown("**トライアド構造 (Triadic Census - 3ノード関係)**")
    try:
        triadic_census_result = nx.triadic_census(G_to_analyze)
        st.write("  - 3ノード間の関係性の種類と数:")
        df_triads = pd.DataFrame(list(triadic_census_result.items()), columns=['Motif ID', 'Count'])
        st.dataframe(df_triads[df_triads['Count'] > 0].sort_values(by='Count', ascending=False)) # 数が0のものは非表示
        st.caption("""
            Motif IDの例:
            - `003`: 3ノード間にエッジなし
            - `012`: A→B (他エッジなし)
            - `102`: A→B, B→A (相互、他なし)
            - `021D`: A←B→C (Bが共通のフォロー先)
            - `021U`: A→B←C (Bが共通のフォロワー)
            - `021C`: A→B→C (Bが中継)
            - `111D`: A→B←C, A→C
            - `111U`: A←B→C, A←C
            - `030T`: A→B, B→C, C→A (3サイクル)
            - `030C`: A←B→C, A←C→B (AがB,C両方からフォローされ、B,C間エッジなし)
            - `201`: A→B, B→A, C→A
            - `120D`: A→B, B→C, A→C, C→B
            - `120U`: A←B, B←C, A←C, C←A
            - `120C`: A→B, B→C, C→A, A→C (030T + A→C)
            - `210`: A→B, B→A, A→C, C→A, B→C (or C→B)
            - `300`: A,B,C 全員相互フォロー (完全グラフ)
            (詳細はNetworkXドキュメント参照)
        """)
    except Exception as e:
        st.warning(f"- トライアド分析の計算中にエラー: {e}")
elif G_to_analyze.is_directed() and num_nodes < 3 :
     st.markdown("**トライアド構造 (Triadic Census - 3ノード関係)**")
     st.write("- トライアド構造: ノード数が3未満のため計算できません。")
elif G_to_analyze.is_directed() and num_nodes >=300:
     st.markdown("**トライアド構造 (Triadic Census - 3ノード関係)**")
     st.write("- トライアド構造: グラフサイズが大きいため計算をスキップ。")


st.markdown("---")
st.caption("凡例: N=ノード数. 一部の計算（平均最短経路長、媒介中心性、トライアド分析など）はグラフのサイズによって非常に時間がかかるため、大きなグラフではスキップまたは近似計算、もしくは一部の連結成分のみで計算されています。")