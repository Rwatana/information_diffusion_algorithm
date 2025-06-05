import streamlit as st
import networkx as nx
import random
# import matplotlib.pyplot as plt # グラフ可視化（最終状態）に使う場合
import sys
import os
import pandas as pd # Pandas をインポート
import json
import pandas as pd # Pandas をインポート（データフレーム操作用）
from streamlit_agraph import agraph, Node, Edge, Config # agraph をインポート
from datetime import datetime

# --- パス設定 ---
current_dir = os.path.dirname(__file__)
streamlit_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root_dir = os.path.abspath(os.path.join(streamlit_dir, '..'))
sys.path.append(project_root_dir)

from datagen.data_utils import simulate_ic # 修正された simulate_ic をインポート

# 保存用ディレクトリ
SAVE_DIR = os.path.join(streamlit_dir, "saved_toy_experiments_v2")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.set_page_config(layout="wide", page_title="Toy 伝播実験 (エッジ確率対応)")
st.title("Toy 伝播実験 (ICモデル - エッジごとの確率使用)")

# --- Helper Functions ---
def save_experiment_settings_simple(graph_simple_data, propagation_probability_default, base_name="experiment"):
    """グラフ構造（重みなし）とデフォルト伝播確率を保存"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{base_name}_{timestamp}"
    full_folder_path = os.path.join(SAVE_DIR, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    settings_data = {
        "default_propagation_probability": propagation_probability_default,
        "graph_node_count": len(graph_simple_data["nodes"]),
        "graph_edge_count": len(graph_simple_data["edges"])
    }
    with open(os.path.join(full_folder_path, "graph_simple.json"), "w") as f_graph:
        json.dump(graph_simple_data, f_graph)
    with open(os.path.join(full_folder_path, "settings_simple.json"), "w") as f_settings:
        json.dump(settings_data, f_settings)
    return full_folder_path

def load_experiment_settings_simple(folder_path):
    """保存されたグラフ構造（重みなし）とデフォルト伝播確率をロード"""
    try:
        with open(os.path.join(folder_path, "graph_simple.json"), "r") as f_graph:
            graph_data = json.load(f_graph)
        with open(os.path.join(folder_path, "settings_simple.json"), "r") as f_settings:
            settings_data = json.load(f_settings)

        G_loaded = nx.DiGraph() if graph_data.get("directed", True) else nx.Graph()
        G_loaded.add_nodes_from(graph_data["nodes"])
        G_loaded.add_edges_from(graph_data["edges"]) # 重みはここではロードしない
        return G_loaded, settings_data["default_propagation_probability"]
    except FileNotFoundError:
        st.error(f"エラー: 必要なファイルが見つかりません ({folder_path})。")
        return None, None
    except Exception as e:
        st.error(f"実験設定のロード中にエラーが発生しました: {e}")
        return None, None

def get_saved_experiments_simple():
    if not os.path.exists(SAVE_DIR): return []
    return sorted([d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d)) and \
                   os.path.exists(os.path.join(SAVE_DIR, d, "graph_simple.json"))], reverse=True)


# --- サイドバー ---
st.sidebar.header("実験設定")

# セッションステート初期化
if 'tp_current_graph_sim' not in st.session_state: st.session_state.tp_current_graph_sim = None
if 'tp_default_prop_prob_sim' not in st.session_state: st.session_state.tp_default_prop_prob_sim = 0.1
if 'tp_loaded_experiment_name_sim' not in st.session_state: st.session_state.tp_loaded_experiment_name_sim = "未選択"
if 'tp_graph_has_edge_weights' not in st.session_state: st.session_state.tp_graph_has_edge_weights = False


experiment_action_sim = st.sidebar.radio(
    "実験に使用するグラフ:",
    ("「グラフ可視化」タブのグラフを使用", "保存された実験をロード"),
    key="tp_exp_action_choice_sim", index=0
)

if experiment_action_sim == "「グラフ可視化」タブのグラフを使用":
    # 1_graph_visualization.py で使用されているセッションキーに合わせる
    graph_key_gv_weighted = 'gv_graph_with_weights_static' # 修正
    graph_key_gv_base = 'gv_graph_base_static'         # 修正

    graph_to_apply = None
    source_name = ""
    has_weights_applied = False

    if graph_key_gv_weighted in st.session_state and st.session_state.get(graph_key_gv_weighted) is not None:
        graph_to_apply = st.session_state.get(graph_key_gv_weighted)
        source_name = "「グラフ可視化」タブのグラフ (エッジ確率割り当て済み)"
        has_weights_applied = True
    elif graph_key_gv_base in st.session_state and st.session_state.get(graph_key_gv_base) is not None: # 修正
        graph_to_apply = st.session_state.get(graph_key_gv_base)
        source_name = "「グラフ可視化」タブのグラフ (デフォルト確率使用)"
        has_weights_applied = False
    else:
        st.sidebar.warning("「グラフ可視化」タブでグラフが生成されていません。")

    if graph_to_apply and graph_to_apply.number_of_nodes() > 0:
        if st.sidebar.button("このグラフを実験に適用", key="tp_apply_gv_graph_button_sim"):
            st.session_state.tp_current_graph_sim = graph_to_apply
            st.session_state.tp_default_prop_prob_sim = 0.1 # デフォルト確率 (エッジ重みがない場合用)
            st.session_state.tp_loaded_experiment_name_sim = source_name
            st.session_state.tp_graph_has_edge_weights = has_weights_applied
            st.sidebar.success(f"「{source_name}」を適用しました。")
            st.rerun()
        if st.session_state.tp_current_graph_sim is None or \
           st.session_state.tp_loaded_experiment_name_sim != source_name:
            st.sidebar.info("上のボタンを押してグラフをこの実験に適用してください。")
    elif graph_to_apply is None and (graph_key_gv_weighted in st.session_state or graph_key_gv_base in st.session_state) :
         st.sidebar.warning("「グラフ可視化」タブのグラフは空か、ノードがありません。")


elif experiment_action_sim == "保存された実験をロード":
    saved_experiments_list = get_saved_experiments_simple()
    if not saved_experiments_list:
        st.sidebar.info("保存された実験はありません。")
    else:
        selected_exp_folder_load = st.sidebar.selectbox(
            "ロードする実験フォルダを選択:", saved_experiments_list, index=None,
            placeholder="実験フォルダを選択...", key="tp_select_saved_exp_folder_sim"
        )
        if selected_exp_folder_load and st.sidebar.button("選択した実験をロード", key="tp_load_selected_exp_button_sim"):
            G_loaded_sim, prop_prob_loaded_sim = load_experiment_settings_simple(os.path.join(SAVE_DIR, selected_exp_folder_load))
            if G_loaded_sim is not None:
                st.session_state.tp_current_graph_sim = G_loaded_sim
                st.session_state.tp_default_prop_prob_sim = prop_prob_loaded_sim
                st.session_state.tp_loaded_experiment_name_sim = selected_exp_folder_load
                st.session_state.tp_graph_has_edge_weights = False # 保存データはエッジ重みなしと仮定
                st.sidebar.success(f"実験「{selected_exp_folder_load}」をロードしました。デフォルト伝播確率もロードされました。")
                st.rerun()

G_sim = st.session_state.get('tp_current_graph_sim')
default_prop_prob_for_slider = st.session_state.get('tp_default_prop_prob_sim', 0.1)
current_experiment_name_sim = st.session_state.get('tp_loaded_experiment_name_sim', "未選択")
graph_has_edge_weights_sim = st.session_state.get('tp_graph_has_edge_weights', False)

if G_sim is None:
    st.info("実験対象のグラフを選択またはロードしてください。")
    st.stop()

st.subheader(f"現在のグラフ: {current_experiment_name_sim}")
st.write(f"ノード数: {G_sim.number_of_nodes()}, エッジ数: {G_sim.number_of_edges()}, 有向: {G_sim.is_directed()}")
if graph_has_edge_weights_sim:
    st.success("このグラフはエッジごとの伝播確率が設定されています。シミュレーションではそれが優先されます。")
else:
    st.info("このグラフはエッジごとの伝播確率が設定されていません。下の「デフォルト活性化確率」が使用されます。")


st.sidebar.markdown("---")
st.sidebar.subheader("ICモデル デフォルト設定")
default_activation_probability_input = st.sidebar.slider(
    "デフォルト活性化確率 (p):", 0.01, 1.0, float(default_prop_prob_for_slider), 0.01,
    key="tp_default_activation_prob_slider_sim",
    help="グラフのエッジに個別の伝播確率が設定されていない場合、または保存/ロード機能で使用されるデフォルト値です。"
)

exp_save_name_base_input_sim = st.sidebar.text_input("保存名ベース (任意):", value="toy_exp_simple", key="tp_save_name_input_sim")
if st.sidebar.button("現在のグラフと「デフォルト」確率を保存", key="tp_save_current_exp_button_sim"):
    if G_sim is not None:
        simple_graph_data = {
            "nodes": list(G_sim.nodes()),
            "edges": list(G_sim.edges()),
            "directed": G_sim.is_directed()
        }
        saved_path_simple = save_experiment_settings_simple(simple_graph_data, default_activation_probability_input, exp_save_name_base_input_sim)
        st.sidebar.success(f"設定を保存しました: {os.path.basename(saved_path_simple)}")
        st.rerun()
    else:
        st.sidebar.error("保存するグラフがありません。")

st.sidebar.markdown("---")
st.sidebar.subheader("シミュレーション実行")
initial_nodes_options_sim = sorted(list(G_sim.nodes()))
default_initial_nodes_sim = []
if initial_nodes_options_sim:
    default_initial_nodes_sim = random.sample(initial_nodes_options_sim, min(3, len(initial_nodes_options_sim)))

initial_nodes_selected_sim = st.sidebar.multiselect(
    "初期活性化ノードを選択:", initial_nodes_options_sim, default=default_initial_nodes_sim, key="tp_initial_nodes_multiselect_sim"
)

if not initial_nodes_selected_sim and G_sim.number_of_nodes() > 0 :
    st.sidebar.warning("少なくとも1つの初期活性化ノードを選択してください。")

if st.sidebar.button("伝播実験開始", key="tp_run_propagation_main_button_sim"):
    if not initial_nodes_selected_sim:
        st.error("初期活性化ノードが選択されていません。")
    elif G_sim is None:
        st.error("実験するグラフがロードされていません。")
    else:
        final_default_activation_probability = default_activation_probability_input

        st.session_state['tp_sim_history_run_main'] = None
        st.session_state['tp_sim_final_activated_run_main'] = None

        st.subheader("伝播の様子")
        if graph_has_edge_weights_sim:
            st.write(f"情報: エッジごとの伝播確率を使用します (デフォルト確率 {final_default_activation_probability:.2f} は重みなしエッジ用)。")
        else:
            st.write(f"使用したデフォルト伝播確率: {final_default_activation_probability:.2f}")

        activated_n_set, prop_log_list = simulate_ic(G_sim, initial_nodes_selected_sim, final_default_activation_probability)
        
        st.session_state['tp_sim_final_activated_run_main'] = activated_n_set
        simulation_history = []
        
        # 初期状態 (ステップ0) のログを作成
        simulation_history.append({
            "step": 0,
            "newly_activated": sorted(list(set(initial_nodes_selected_sim))),
            "total_activated_count": len(set(initial_nodes_selected_sim)),
            "all_activated_nodes_snapshot": sorted(list(set(initial_nodes_selected_sim)))
        })
        st.write(f"初期活性化ノード (ステップ 0): {simulation_history[0]['newly_activated']}")
        st.write(f"現在の活性化ノード総数: {simulation_history[0]['total_activated_count']}")
        st.write("---")

        if prop_log_list:
            log_df_for_history = pd.DataFrame(prop_log_list)
            max_step_hist = int(log_df_for_history['step'].max()) if not log_df_for_history.empty else 0
            
            temp_activated_for_hist = set(initial_nodes_selected_sim)

            for s_idx in range(1, max_step_hist + 1):
                newly_this_s_nodes = set(log_df_for_history[log_df_for_history['step'] == s_idx]['target'].unique())
                newly_this_s_display = sorted(list(newly_this_s_nodes)) # 表示用
                temp_activated_for_hist.update(newly_this_s_nodes)
                
                simulation_history.append({
                    "step": s_idx,
                    "newly_activated": newly_this_s_display,
                    "total_activated_count": len(temp_activated_for_hist),
                    "all_activated_nodes_snapshot": sorted(list(temp_activated_for_hist))
                })
                st.write(f"**ステップ {s_idx}:**")
                if newly_this_s_display: st.write(f"  新たに活性化したノード: {newly_this_s_display}")
                else: st.write("  新たに活性化したノード: なし")
                st.write(f"  現在の活性化ノード総数: {len(temp_activated_for_hist)}")
                st.write("---")
            
            if max_step_hist >= 0 : # 伝播があったか、初期シードだけでも
                 st.subheader("実験終了")
                 st.write(f"**最終的な活性化ノード数: {len(activated_n_set)}**")

        elif initial_nodes_selected_sim:
            st.subheader("初期ノードのみ活性化 (伝播なし)")
            st.write(f"**最終的な活性化ノード数: {len(initial_nodes_selected_sim)}**")

        st.session_state['tp_sim_history_run_main'] = simulation_history


# --- 実験結果の可視化 (最終状態) ---
if 'tp_sim_history_run_main' in st.session_state and st.session_state['tp_sim_history_run_main']:
    st.subheader("最終状態のグラフ可視化")
    final_activated_nodes_viz_main = st.session_state.get('tp_sim_final_activated_run_main', set())
    initial_nodes_viz_set_main = set()
    if st.session_state['tp_sim_history_run_main'] and len(st.session_state['tp_sim_history_run_main']) > 0:
        initial_nodes_viz_set_main = set(st.session_state['tp_sim_history_run_main'][0].get('newly_activated', []))

    if G_sim:
        nodes_vis_main = []
        for node_id_viz in G_sim.nodes(): # イテレータ変数名を変更
            color, size, shape = "#E0E0E0", 12, "dot"
            if node_id_viz in final_activated_nodes_viz_main:
                if node_id_viz in initial_nodes_viz_set_main: color, size, shape = "red", 20, "star"
                else: color, size = "orange", 15
            nodes_vis_main.append(Node(id=str(node_id_viz), label=str(node_id_viz), color=color, size=size, shape=shape))

        edges_vis_main = []
        # 最終状態の活性化エッジを強調表示（簡易版）
        # simulate_icが返すprop_log_listから活性化したエッジを取得するのがより正確
        activated_edges_from_log = set()
        if st.session_state.get('tp_sim_history_run_main'): # 履歴があるか確認
            full_prop_log = []
            for hist_entry in st.session_state['tp_sim_history_run_main']:
                # simulate_ic のログ形式に合わせてアクセスする必要がある
                # ここでは simulate_ic が返す prop_log_list を直接セッションに保存し、それを使うべき
                # 今回は簡易的に、両端が最終的にアクティブなエッジの色を変える
                pass # 以下のループで処理

        for u_edge, v_edge in G_sim.edges(): # イテレータ変数名を変更
            edge_color_viz = "#E0E0E0"
            # 簡易的な強調：最終的に両端が活性化していれば色を変える
            if u_edge in final_activated_nodes_viz_main and v_edge in final_activated_nodes_viz_main:
                edge_color_viz = "#B0C4DE" # 薄い青
            edges_vis_main.append(Edge(source=str(u_edge),target=str(v_edge),color=edge_color_viz))

        config_viz_main = Config(width="100%", height=600, directed=G_sim.is_directed(), physics=False)
        if nodes_vis_main:
            agraph(nodes=nodes_vis_main, edges=edges_vis_main, config=config_viz_main)
    else:
        st.warning("グラフがロードされていません。")
