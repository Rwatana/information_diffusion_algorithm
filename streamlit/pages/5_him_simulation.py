import streamlit as st
import pandas as pd
import random
import sys
import os
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import torch # HIMモデルがPyTorchを使用する場合

# --- パス設定 ---
current_dir = os.path.dirname(__file__)
streamlit_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root_dir = os.path.abspath(os.path.join(streamlit_dir, '..'))
sys.path.append(project_root_dir)

from datagen.data_utils import simulate_ic, generate_propagations # generate_propagationsもインポート
from him_full.him_model import HIMModel # HIMモデルクラスをインポート
from him_full.seed_selection import adaptive_sliding_window # シード選択アルゴリズム

st.set_page_config(layout="wide", page_title="HIM シミュレーション")
st.title("HIM ベースの影響最大化シミュレーション")

graph_key = 'gv_graph' # グラフ可視化ページで設定されたグラフのキー

if graph_key not in st.session_state or st.session_state[graph_key] is None:
    st.warning("最初に「グラフ可視化」タブでグラフを生成してください。")
    st.stop()

G = st.session_state[graph_key]
num_graph_nodes = G.number_of_nodes()

if num_graph_nodes == 0:
    st.warning("グラフにノードがありません。「グラフ可視化」タブで再生成してください。")
    st.stop()

# --- サイドバー設定 ---
st.sidebar.header("HIM シミュレーション設定")
max_seeds = num_graph_nodes
num_seeds = st.sidebar.slider("シード数 (k)", 1, max_seeds, min(10, max_seeds), key="him_num_seeds")
propagation_prob_sim = st.sidebar.slider("伝播確率 (p) for Simulation", 0.01, 1.0, 0.1, 0.01, key="him_prop_prob")

st.sidebar.markdown("---")
st.sidebar.caption("HIMモデル学習設定")
him_dim = st.sidebar.slider("埋め込み次元 (dim)", 8, 64, 32, step=8, key="him_dim_param")
him_epochs = st.sidebar.slider("学習エポック数", 1, 500, 10, key="him_epochs_param") # Streamlit上では軽めに
him_beta = st.sidebar.slider("Adaptive Sliding Window Beta", 0.1, 5.0, 1.0, step=0.1, key="him_beta_param")
# 伝播インスタンスを簡易生成するかどうかのオプション
generate_dummy_props = st.sidebar.checkbox("簡易伝播インスタンスを学習に使用する", value=False, key="him_gen_props")
num_dummy_prop_instances = 0
if generate_dummy_props:
    num_dummy_prop_instances = st.sidebar.slider("簡易伝播インスタンス数", 1, 20, 5, key="him_num_dummy_props")


run_simulation_button = st.sidebar.button("HIMモデル学習 & シミュレーション実行", key="him_run_sim_button_actual")

# --- メイン処理 ---
if run_simulation_button:
    # セッションステートの初期化
    for key in ['him_seeds_selected', 'him_simulation_log', 'him_final_activated_nodes', 'him_stepwise_cumulative_nodes', 'him_model_trained', 'him_embeddings']:
        if key in st.session_state:
            del st.session_state[key]

    st.subheader("HIMモデル学習 & シード選択")
    
    # 1. (オプション) 簡易伝播インスタンスの生成
    propagations_for_training = []
    if generate_dummy_props and num_dummy_prop_instances > 0:
        with st.spinner(f"{num_dummy_prop_instances}件の簡易伝播インスタンスを生成中..."):
            # generate_propagations に渡すseed_countはグラフノード数より小さくする必要がある
            prop_seed_count = min(5, num_graph_nodes // 10, num_seeds) # 適当な値
            if prop_seed_count < 1 and num_graph_nodes > 0 : prop_seed_count = 1

            if num_graph_nodes > 0 and prop_seed_count > 0 :
                 propagations_for_training = generate_propagations(
                     G,
                     seed_count=prop_seed_count,
                     num_instances=num_dummy_prop_instances,
                     ic_prob=0.05 # 固定の確率で簡易生成
                 )
                 st.write(f"{len(propagations_for_training)}件の伝播インスタンスを生成しました。")
            else:
                 st.write("ノード数が少ないため、伝播インスタンスは生成しませんでした。")


    # 2. HIMモデルの学習
    seeds = None # シード変数を初期化
    if num_graph_nodes > 0 : # ノードが存在する場合のみ学習・シード選択
        with st.spinner(f"HIMモデルを学習中 (dim={him_dim}, epochs={him_epochs})... これは時間がかかる場合があります。"):
            try:
                model = HIMModel(num_nodes=num_graph_nodes, dim=him_dim)
                model.fit(G, propagations_for_training, epochs=him_epochs, verbose=False) # verbose=False でStreamlitの出力を簡潔に
                st.session_state['him_model_trained'] = True # モデル学習完了フラグ
                st.session_state['him_embeddings'] = model.embeddings.detach().cpu() # CPUに移動して保存
                st.success("HIMモデルの学習が完了しました。")
            except Exception as e:
                st.error(f"HIMモデルの学習中にエラーが発生しました: {e}")
                st.session_state['him_model_trained'] = False

        # 3. Adaptive Sliding Windowによるシード選択
        if st.session_state.get('him_model_trained'):
            with st.spinner("学習済み埋め込みを使用してシードを選択中..."):
                try:
                    emb = st.session_state['him_embeddings']
                    # adaptive_sliding_windowの引数を確認し、正しく渡す
                    # G, emb, k, beta が主要な引数と仮定
                    seeds = adaptive_sliding_window(G, emb, k=num_seeds, beta=him_beta)
                    st.success("シード選択が完了しました。")
                except Exception as e:
                    st.error(f"シード選択中にエラーが発生しました: {e}")
                    seeds = None # エラー時はシードをNoneに
        else:
            st.warning("HIMモデルの学習に失敗したため、シード選択を実行できません。")
            seeds = None
    else: # グラフにノードがない場合
        st.error("グラフにノードがないため、HIMモデルの処理を実行できません。")
        seeds = None


    if seeds:
        st.write(f"選択されたシードノード ({len(seeds)}個): {sorted(seeds)}")
        st.session_state['him_seeds_selected'] = seeds

        st.subheader("影響伝播シミュレーション結果 (HIM)")
        with st.spinner("伝播シミュレーションを実行中..."):
            final_activated_nodes, raw_log = simulate_ic(G, seeds, propagation_prob_sim)
        
        st.session_state['him_simulation_log'] = raw_log
        st.session_state['him_final_activated_nodes'] = final_activated_nodes
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
        st.session_state['him_stepwise_cumulative_nodes'] = stepwise_cumulative
    elif run_simulation_button: # ボタンは押されたがシードが選択できなかった場合
        st.error("HIMによるシード選択に失敗しました。シミュレーションを実行できません。")


# --- 結果表示 (前回のコードと同様のロジック、キー名は him_ で統一) ---
if st.session_state.get('him_seeds_selected'):
    st.markdown("---")
    st.header("HIM シミュレーション結果表示")
    st.write(f"選択されたシード (HIM): {sorted(st.session_state['him_seeds_selected'])}")

    raw_log_df_display = pd.DataFrame(st.session_state.get('him_simulation_log', []))
    initial_seeds_display = st.session_state.get('him_seeds_selected', []) # シミュレーションに使ったシード

    if not raw_log_df_display.empty or initial_seeds_display:
        st.write("ステップごとに新たに活性化されたノード:")
        # (以前のコードから流用した新規活性化ノード表示ロジック)
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
            if isinstance(nodes, list) and nodes: # ノードリストが空でない場合のみ
                 padded_nodes = nodes + [pd.NA] * (max_len_disp - len(nodes))
                 display_data_df_disp[f"Step {step}"] = padded_nodes
        
        if display_data_df_disp:
            st.dataframe(pd.DataFrame(display_data_df_disp).astype(str).replace('<NA>', ''), height=200, use_container_width=True)
        elif initial_seeds_display: # 初期シードのみの場合
             st.dataframe(pd.DataFrame({ "Step 0": sorted(list(initial_seeds_display))}).astype(str), height=200, use_container_width=True)


        st.write(f"最終活性化ノード数: {len(st.session_state.get('him_final_activated_nodes', []))}")

        st.subheader("ステップごとのグラフ状態可視化 (HIM)")
        stepwise_cumulative_map_disp = st.session_state.get('him_stepwise_cumulative_nodes', {})
        if stepwise_cumulative_map_disp:
            max_slider_step_disp = max(stepwise_cumulative_map_disp.keys()) if stepwise_cumulative_map_disp else 0
            
            selected_step_val = 0
            if max_slider_step_disp > 0:
                selected_step_val = st.slider("表示ステップ選択", 0, max_slider_step_disp, max_slider_step_disp, key="him_step_slider_viz")
            elif 0 in stepwise_cumulative_map_disp :
                st.write("ステップ 0 (初期シード状態) のみ表示します。")
            # else:
            #     st.info("表示できる伝播ステップがありません。") # シミュレーション未実行時はここに到達しない想定


            nodes_active_now = stepwise_cumulative_map_disp.get(selected_step_val, set())
            newly_active_now = set()
            if selected_step_val > 0: # selected_step_valが定義されていれば
                nodes_active_prev = stepwise_cumulative_map_disp.get(selected_step_val - 1, set())
                newly_active_now = nodes_active_now - nodes_active_prev

            nodes_v, edges_v = [], []
            # (以前のコードから流用したノード・エッジ可視化ロジック)
            for node_id in G.nodes():
                color, size, shape, b_width = "#E0E0E0", 12, "dot", 0
                if node_id in nodes_active_now:
                    if node_id in initial_seeds_display: color, size, shape = "red", 25, "star"
                    elif node_id in newly_active_now: color, size, b_width = "orange", 20, 2
                    else: color, size = "#FFD700", 18
                nodes_v.append(Node(id=str(node_id), label=str(node_id), color=color, size=size, shape=shape, borderWidth=b_width))

            for u, v_target_node in G.edges():
                ec, ew = "#E0E0E0", 1
                active_edge = False
                if not raw_log_df_display.empty: # raw_log_df_display が空でないことを確認
                     if not raw_log_df_display[(raw_log_df_display['source']==u) & (raw_log_df_display['target']==v_target_node) & (raw_log_df_display['step'] <= selected_step_val)].empty:
                        active_edge = True
                if active_edge and u in nodes_active_now and v_target_node in nodes_active_now: ec, ew = "blue", 2.5
                elif u in nodes_active_now and v_target_node in nodes_active_now: ec = "#B0C4DE"
                edges_v.append(Edge(source=str(u), target=str(v_target_node), color=ec, width=ew))
            
            config_viz = Config(width="100%", height=700, directed=G.is_directed(), physics=False,
                                nodes={'font': {'size': 10}}, edges={'smooth': {'type': 'continuous'}})
            if nodes_v:
                st.write(f"**ステップ {selected_step_val} の状態 (HIM):**")
                agraph(nodes=nodes_v, edges=edges_v, config=config_viz)
        # else: # stepwise_cumulative_map_disp が空の場合
        #     st.info("シミュレーションデータがありません。") # シミュレーション未実行時はここに到達しない想定

elif run_simulation_button: # ボタンは押されたがシードが選択/保存されなかった場合
    st.error("HIMによるシミュレーションの実行に失敗しました。")