import streamlit as st
import pandas as pd
import random
import sys
import os
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import json
from datetime import datetime

# --- パス設定とモジュールのインポート ---
# このスクリプト(2_propagation_log.py)の場所を基準にプロジェクトルートを特定
# .../streamlit/pages/ -> .../streamlit/ -> .../ (Project Root)
try:
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_dir = os.path.abspath(os.path.join(current_file_dir, '..'))
    project_root = os.path.abspath(os.path.join(streamlit_app_dir, '..'))
    
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    from datagen.data_utils import simulate_ic

except (ImportError, ModuleNotFoundError):
    st.error("エラー: `datagen.data_utils` モジュールが見つかりません。")
    st.info(f"プロジェクトルートを `{project_root}` に設定しようとしましたが、失敗しました。ディレクトリ構造を確認してください。")
    st.stop()


# --- 定数とディレクトリ設定 (重要：パスを修正) ---
# このスクリプトと同じ階層にある`saved_graphs`を保存場所とする
SAVE_DIR_NAME = "saved_graphs"
SAVE_DIR_PATH = os.path.join(current_file_dir, SAVE_DIR_NAME) 

if not os.path.exists(SAVE_DIR_PATH):
    st.error(f"保存ディレクトリが見つかりません: {SAVE_DIR_PATH}")
    st.info("`1_graph_visualization.py` などのページでグラフを保存すると、自動的に作成されます。")
    st.stop()


# --- ヘルパー関数 ---
# --- ヘルパー関数 (修正版) ---

def load_graph_from_json(folder_name):
    """フォルダ名を受け取り、その中のgraph_data.jsonを読み込みます。"""
    filepath = os.path.join(SAVE_DIR_PATH, folder_name, 'graph_data.json')
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # グラフデータに 'metadata' が含まれている可能性があるため、'links'や'nodes'キーがあるか確認
        if 'nodes' in data and 'links' in data:
            return nx.node_link_graph(data)
        else:
            st.error(f"エラー: {folder_name} のJSONファイルは有効なグラフデータ形式ではありません。")
            return None
    except FileNotFoundError:
        st.error(f"エラー: {filepath} が見つかりません。")
        return None
    except Exception as e:
        st.error(f"グラフ読込エラー: {e}")
        return None

def get_saved_graph_files():
    """保存されているグラフの「フォルダ」リストを取得します。"""
    if not os.path.exists(SAVE_DIR_PATH): return []
    
    # saved_graphs内の各項目が「フォルダ」であるかを確認してリスト化する
    return sorted([
        d for d in os.listdir(SAVE_DIR_PATH)
        if os.path.isdir(os.path.join(SAVE_DIR_PATH, d))
    ], reverse=True)

# --- セッションステートの初期化 (ページ固有キーを使用) ---
if 'prop_log_graph' not in st.session_state:
    st.session_state.prop_log_graph = None
if 'prop_log_seed_nodes_str' not in st.session_state:
    st.session_state.prop_log_seed_nodes_str = ""
if 'prop_log_simulation_results' not in st.session_state:
    st.session_state.prop_log_simulation_results = None

# --- サイドバー ---
st.sidebar.header("伝播ログ分析")

# 1. 保存済みグラフ読み込み
st.sidebar.subheader("1. グラフを選択")
st.sidebar.caption(f"グラフ保存場所: `pages/saved_graphs`")

saved_graph_files = get_saved_graph_files()

if not saved_graph_files:
    st.sidebar.warning("保存されているグラフがありません。")
else:
    # `index=0`とプレースホルダーで、最初は何も選択されていない状態にする
    selected_file_for_load = st.sidebar.selectbox(
        "読み込むグラフファイルを選択",
        options=saved_graph_files,
        index=None,
        placeholder="ファイルを選択してください",
        key="prop_log_load_selector"
    )

    if st.sidebar.button("選択したグラフを読み込み", key="prop_log_load_btn"):
        if selected_file_for_load:
            # グラフを読み込み、このページ用のセッションステートに保存
            graph = load_graph_from_json(selected_file_for_load)
            st.session_state.prop_log_graph = graph
            # シミュレーション関連の状態をリセット
            st.session_state.prop_log_seed_nodes_str = ""
            st.session_state.prop_log_simulation_results = None
            st.toast(f"{selected_file_for_load} を読み込みました。", icon="✅")
            st.rerun()
        else:
            st.sidebar.warning("読み込むファイルを選択してください。")

# 2. 伝播シミュレーション設定
st.sidebar.subheader("2. シミュレーションを実行")

active_graph = st.session_state.get('prop_log_graph')

# グラフが読み込まれている場合のみ、シミュレーション設定を表示
if active_graph:
    # シードノードの自動提案
    if not st.session_state.prop_log_seed_nodes_str and active_graph.number_of_nodes() > 0:
        try:
            num_seeds = min(3, active_graph.number_of_nodes())
            nodes = list(active_graph.nodes())
            st.session_state.prop_log_seed_nodes_str = ", ".join(map(str, random.sample(nodes, num_seeds)))
        except ValueError:
            st.session_state.prop_log_seed_nodes_str = ""

    seed_nodes_str_input = st.sidebar.text_input(
        "シードノード (カンマ区切り)",
        value=st.session_state.prop_log_seed_nodes_str,
        key="prop_log_seed_input"
    )
    st.session_state.prop_log_seed_nodes_str = seed_nodes_str_input

    # シードノードの検証
    parsed_valid_seed_nodes = []
    if seed_nodes_str_input.strip():
        try:
            raw_seeds = [s.strip() for s in seed_nodes_str_input.split(',') if s.strip()]
            potential_seeds = [int(s) for s in raw_seeds]
            parsed_valid_seed_nodes = [s for s in potential_seeds if s in active_graph.nodes()]
            if len(raw_seeds) != len(parsed_valid_seed_nodes):
                st.sidebar.warning("一部のシードノードはグラフ内に存在しません。")
        except ValueError:
            st.sidebar.error("シードノードはカンマ区切りの数値で入力してください。")
    
    run_sim_btn_disabled = not parsed_valid_seed_nodes
    if st.sidebar.button("伝播シミュレーション実行", key="prop_log_run_sim_btn", disabled=run_sim_btn_disabled):
        final_nodes, log = simulate_ic(active_graph, set(parsed_valid_seed_nodes))
        
        cumulative_activated = {0: set(parsed_valid_seed_nodes)}
        current_total = set(parsed_valid_seed_nodes)
        if log:
            log_df = pd.DataFrame(log)
            max_step = int(log_df['step'].max()) if not log_df.empty else 0
            for i in range(1, max_step + 1):
                newly_activated = set(log_df[log_df['step'] == i]['target'])
                current_total.update(newly_activated)
                cumulative_activated[i] = current_total.copy()
        
        st.session_state.prop_log_simulation_results = {
            "seeds": set(parsed_valid_seed_nodes),
            "log": log,
            "final_activated": final_nodes,
            "cumulative_activated": cumulative_activated
        }
        st.toast(f"最終活性化ノード数: {len(final_nodes)}", icon="🎯")
        st.rerun()
else:
    st.sidebar.info("グラフを読み込むと、シミュレーション設定が表示されます。")


# --- メインエリア ---
st.title("影響伝播シミュレーションとログ分析")

if not active_graph:
    st.info("サイドバーから分析対象のグラフを読み込んでください。")
    st.stop()

st.header("現在のグラフ")
main_cols = st.columns(2)
main_cols[0].metric("ノード数", active_graph.number_of_nodes())
main_cols[1].metric("エッジ数", active_graph.number_of_edges())

# --- シミュレーション結果の表示 ---
simulation_results = st.session_state.get('prop_log_simulation_results')
if simulation_results:
    st.header("影響伝播結果")
    
    log = simulation_results['log']
    seeds = simulation_results['seeds']
    final_activated = simulation_results['final_activated']
    cumulative_map = simulation_results['cumulative_activated']

    # ステップごとのスライダーと可視化
    if cumulative_map:
        max_step = max(cumulative_map.keys())
        chosen_step = 0
        if max_step > 0:
            chosen_step = st.slider("表示するステップを選択:", 0, max_step, max_step, key="prop_log_step_slider")
        
        st.subheader(f"ステップ {chosen_step} の状態")
        
        nodes_active_at_step = cumulative_map[chosen_step]
        
        viz_nodes_prop = []
        for node in active_graph.nodes():
            color, size, shape = "#D3D3D3", 12, "dot" # Default
            if node in nodes_active_at_step:
                if node in seeds:
                    color, size, shape = "red", 25, "star"
                else:
                    color, size = "orange", 18
            viz_nodes_prop.append(Node(id=str(node), label=str(node), color=color, size=size, shape=shape))

        viz_edges_prop = []
        log_df = pd.DataFrame(log)
        for u, v, data in active_graph.edges(data=True):
            edge_color, width = "#E0E0E0", 1.0
            is_used = not log_df[(log_df['source'] == u) & (log_df['target'] == v) & (log_df['step'] <= chosen_step)].empty
            if is_used:
                edge_color, width = "blue", 2.5
            elif u in nodes_active_at_step and v not in nodes_active_at_step:
                edge_color = "#FFC0CB" # Pink
            
            viz_edges_prop.append(Edge(source=str(u), target=str(v), label=f"{data.get('weight', 0):.3f}",
                                     color=edge_color, width=width, arrows="to"))

        agraph_config_prop = Config(width="100%", height=700, directed=True, physics=False)
        st.caption("ノード色 - 赤(星): 初期シード, オレンジ: 活性化済み, グレー: 未活性")
        st.caption("エッジ色 - 青: 伝播成功, ピンク: 伝播試行(失敗/未実行), グレー: 未試行")
        agraph(nodes=viz_nodes_prop, edges=viz_edges_prop, config=agraph_config_prop)
else:
    st.info("サイドバーでシードノードを選択し、「伝播シミュレーション実行」ボタンを押してください。")