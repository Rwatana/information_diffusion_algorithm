import streamlit as st
import pandas as pd
import random
import sys
import os
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

# --- パス設定 ---
current_dir = os.path.dirname(__file__)
streamlit_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root_dir = os.path.abspath(os.path.join(streamlit_dir, '..'))
sys.path.append(project_root_dir)

from datagen.data_utils import simulate_ic
from degree_centrality.degree_centrality import select_seeds_by_degree_centrality

st.set_page_config(layout="wide", page_title="次数中心性 シミュレーション")
st.title("次数中心性 ベースの影響最大化シミュレーション")

graph_key = 'gv_graph'

if graph_key not in st.session_state or st.session_state[graph_key] is None:
    st.warning("最初に「グラフ可視化」タブでグラフを生成してください。")
    st.stop()

G = st.session_state[graph_key]
num_graph_nodes = G.number_of_nodes()

if num_graph_nodes == 0:
    st.warning("グラフにノードがありません。「グラフ可視化」タブで再生成してください。")
    st.stop()

# --- サイドバー設定 ---
st.sidebar.header("次数中心性 シミュレーション設定")
degree_type_option = st.sidebar.selectbox("次数タイプ", ["Out-Degree", "In-Degree"], key="dc_degree_type")
use_out_degree_flag = True if degree_type_option == "Out-Degree" else False

max_seeds = num_graph_nodes
num_seeds = st.sidebar.slider("シード数 (k)", 1, max_seeds, min(10, max_seeds), key="dc_num_seeds")
propagation_prob_sim = st.sidebar.slider("伝播確率 (p) for Simulation", 0.01, 1.0, 0.1, 0.01, key="dc_prop_prob")
run_simulation_button = st.sidebar.button(f"{degree_type_option}でシミュレーション実行", key="dc_run_sim_button")

# --- メイン処理 ---
if run_simulation_button:
    st.session_state['dc_seeds_selected'] = None
    st.session_state['dc_simulation_log'] = None
    st.session_state['dc_final_activated_nodes'] = None
    st.session_state['dc_stepwise_cumulative_nodes'] = None

    st.subheader(f"次数中心性 ({degree_type_option}) によるシード選択")
    with st.spinner(f"次数中心性 ({degree_type_option}) でシードを選択中..."):
        seeds = select_seeds_by_degree_centrality(G, num_seeds, use_out_degree=use_out_degree_flag)
    
    if seeds:
        st.write(f"選択されたシードノード ({len(seeds)}個): {sorted(seeds)}")
        st.session_state['dc_seeds_selected'] = seeds

        st.subheader(f"影響伝播シミュレーション結果 (次数中心性 - {degree_type_option})")
        with st.spinner("伝播シミュレーションを実行中..."):
            final_activated_nodes, raw_log = simulate_ic(G, seeds, propagation_prob_sim)
        
        st.session_state['dc_simulation_log'] = raw_log
        st.session_state['dc_final_activated_nodes'] = final_activated_nodes
        st.success(f"シミュレーション完了。最終活性化ノード数: {len(final_activated_nodes)}")

        stepwise_cumulative = {0: set(seeds)}
        current_cumulative = set(seeds)
        if raw_log:
            df_log = pd.DataFrame(raw_log)
            max_step_calc = int(df_log['step'].max()) if not df_log.empty else 0
            for step in range(1, max_step_calc + 1):
                newly_activated = set(df_log[df_log['step'] == step]['target'].unique())
                current_cumulative.update(newly_activated)
                stepwise_cumulative[step] = current_cumulative.copy()
        st.session_state['dc_stepwise_cumulative_nodes'] = stepwise_cumulative
    else:
        st.error(f"次数中心性 ({degree_type_option}) でシードを選択できませんでした。")

# --- 結果表示 (PageRankページと同様のロジック、キー名を dc_ に変更) ---
if st.session_state.get('dc_seeds_selected'):
    st.markdown("---")
    st.header(f"次数中心性 ({degree_type_option}) シミュレーション結果表示") # degree_type_option を取得できるように調整が必要
    # (表示ロジックは PageRank のものとほぼ同じ。セッションキーを dc_ に置き換える)
    # ... (表示部分は PageRank のものを参考に、キー名を dc_ に変えて実装してください) ...
    # 以下、PageRankの表示ロジックをベースにキー名をdc_に変更したものを記載
    st.write(f"選択されたシード (次数中心性 - {st.session_state.get('dc_degree_type_run', degree_type_option)}): {sorted(st.session_state['dc_seeds_selected'])}")
    # 保存された次数タイプを表示するため、ボタンクリック時にセッションにも保存
    if run_simulation_button: st.session_state['dc_degree_type_run'] = degree_type_option


    raw_log_df_display = pd.DataFrame(st.session_state.get('dc_simulation_log', []))
    initial_seeds_display = st.session_state.get('dc_seeds_selected', [])

    if not raw_log_df_display.empty or initial_seeds_display:
        st.write("ステップごとに新たに活性化されたノード:")
        # ... (PageRankと同様の新規活性化ノード表示ロジック, キーはdc_) ...
        stepwise_newly_activated_display = {0: sorted(list(set(initial_seeds_display)))}
        all_nodes_activated_so_far = set(initial_seeds_display)
        max_step_disp = 0
        if not raw_log_df_display.empty:
            max_step_disp = int(raw_log_df_display['step'].max())
        for step_num in range(1, max_step_disp + 1):
            nodes_in_log_this_step = set(raw_log_df_display[raw_log_df_display['step'] == step_num]['target'].unique())
            truly_new_this_step = nodes_in_log_this_step - all_nodes_activated_so_far
            if truly_new_this_step:
                stepwise_newly_activated_display[step_num] = sorted(list(truly_new_this_step))
            all_nodes_activated_so_far.update(nodes_in_log_this_step)
        
        max_len_disp = 0
        if stepwise_newly_activated_display:
            valid_lists_disp = [nodes for nodes in stepwise_newly_activated_display.values() if isinstance(nodes, list) and nodes]
            if valid_lists_disp: max_len_disp = max(len(nodes) for nodes in valid_lists_disp)
        
        display_data_df_disp = {}
        for step, nodes in stepwise_newly_activated_display.items():
            if isinstance(nodes, list) and nodes:
                 padded_nodes = nodes + [pd.NA] * (max_len_disp - len(nodes))
                 display_data_df_disp[f"Step {step}"] = padded_nodes
        
        if display_data_df_disp:
            st.dataframe(pd.DataFrame(display_data_df_disp).astype(str).replace('<NA>', ''), height=200, use_container_width=True)
        elif initial_seeds_display:
             st.dataframe(pd.DataFrame({ "Step 0": sorted(list(initial_seeds_display))}).astype(str), height=200, use_container_width=True)


        st.write(f"最終活性化ノード数: {len(st.session_state.get('dc_final_activated_nodes', []))}")

        st.subheader(f"ステップごとのグラフ状態可視化 (次数中心性 - {st.session_state.get('dc_degree_type_run', degree_type_option)})")
        stepwise_cumulative_map_disp = st.session_state.get('dc_stepwise_cumulative_nodes', {})
        if stepwise_cumulative_map_disp:
            max_slider_step_disp = max(stepwise_cumulative_map_disp.keys()) if stepwise_cumulative_map_disp else 0
            
            selected_step_val = 0
            if max_slider_step_disp > 0:
                selected_step_val = st.slider("表示ステップ選択", 0, max_slider_step_disp, max_slider_step_disp, key="dc_step_slider_viz")
            elif 0 in stepwise_cumulative_map_disp :
                st.write("ステップ 0 (初期シード状態) のみ表示します。")
            # ... (PageRankと同様のグラフ可視化ロジック, キーはdc_) ...
            nodes_active_now = stepwise_cumulative_map_disp.get(selected_step_val, set())
            newly_active_now = set()
            if selected_step_val > 0:
                nodes_active_prev = stepwise_cumulative_map_disp.get(selected_step_val - 1, set())
                newly_active_now = nodes_active_now - nodes_active_prev

            nodes_v, edges_v = [], []
            for node_id in G.nodes():
                color, size, shape, b_width = "#E0E0E0", 12, "dot", 0
                if node_id in nodes_active_now:
                    if node_id in initial_seeds_display: color, size, shape = "red", 25, "star"
                    elif node_id in newly_active_now: color, size, b_width = "orange", 20, 2
                    else: color, size = "#FFD700", 18
                nodes_v.append(Node(id=str(node_id), label=str(node_id), color=color, size=size, shape=shape, borderWidth=b_width))

            for u, v_target_node in G.edges(): # v を v_target_node に変更 (予約語衝突回避)
                ec, ew = "#E0E0E0", 1
                active_edge = False
                if not raw_log_df_display.empty:
                     if not raw_log_df_display[(raw_log_df_display['source']==u) & (raw_log_df_display['target']==v_target_node) & (raw_log_df_display['step'] <= selected_step_val)].empty:
                        active_edge = True
                if active_edge and u in nodes_active_now and v_target_node in nodes_active_now: ec, ew = "blue", 2.5
                elif u in nodes_active_now and v_target_node in nodes_active_now: ec = "#B0C4DE"
                edges_v.append(Edge(source=str(u), target=str(v_target_node), color=ec, width=ew))
            
            config_viz = Config(width="100%", height=700, directed=G.is_directed(), physics=True,
                                nodes={'font': {'size': 10}}, edges={'smooth': {'type': 'continuous'}})
            if nodes_v:
                st.write(f"**ステップ {selected_step_val} の状態 (次数中心性 - {st.session_state.get('dc_degree_type_run', degree_type_option)}):**")
                agraph(nodes=nodes_v, edges=edges_v, config=config_viz)

elif run_simulation_button and not st.session_state.get('dc_seeds_selected'):
    st.error(f"次数中心性 ({degree_type_option}) によるシミュレーションの実行に失敗しました。")