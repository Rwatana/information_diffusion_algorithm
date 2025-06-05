"""
2_propagation_log.py

Streamlit page for displaying propagation logs and visualizing propagation steps.
Located in streamlit/pages/
"""
import streamlit as st
import pandas as pd
import random
import sys
import os
import networkx as nx # NetworkXをインポート
from streamlit_agraph import agraph, Node, Edge, Config # agraph関連をインポート

# 'datagen'フォルダへのパスを追加
current_dir = os.path.dirname(__file__)
streamlit_dir = os.path.abspath(os.path.join(current_dir, '..'))
project_root_dir = os.path.abspath(os.path.join(streamlit_dir, '..'))
sys.path.append(project_root_dir)

from datagen.data_utils import simulate_ic

st.set_page_config(layout="wide", page_title="伝播ログと可視化")
st.title("影響伝播ログとステップごとの可視化")

graph_key = 'gv_graph' # グラフ可視化ページで設定されたグラフのキー

if graph_key not in st.session_state or st.session_state[graph_key] is None:
    st.warning("先に「グラフ可視化」タブでグラフを生成してください。")
    st.stop()

G = st.session_state[graph_key]

st.sidebar.header("伝播シミュレーション設定")

if G.number_of_nodes() == 0:
    st.sidebar.warning("グラフにノードがありません。「グラフ可視化」タブで生成してください。")
    st.stop()

# シードノードのデフォルト値を設定し、セッションステートで管理
if 'pl_seed_nodes_str' not in st.session_state:
    default_seeds_list = []
    if G.number_of_nodes() > 0: # ノードが存在する場合のみサンプル
        default_seeds_list = random.sample(list(G.nodes()), min(3, G.number_of_nodes()))
    st.session_state['pl_seed_nodes_str'] = ", ".join(map(str, default_seeds_list))


seed_nodes_str = st.sidebar.text_input(
    "シードノード (カンマ区切り)",
    value=st.session_state['pl_seed_nodes_str'],
    key="pl_seed_input"
)
st.session_state['pl_seed_nodes_str'] = seed_nodes_str

valid_seed_nodes = []
if seed_nodes_str.strip(): # 入力がある場合のみ処理
    try:
        seed_nodes_input_list = [int(s.strip()) for s in seed_nodes_str.split(',') if s.strip()]
        valid_seed_nodes = [s for s in seed_nodes_input_list if s in G.nodes()]

        if not valid_seed_nodes and seed_nodes_input_list: # 入力はあったが有効なシードがなかった
             st.sidebar.error("入力されたシードノードはグラフ内に存在しません。")
             # st.stop() # ボタンが押されるまでは停止しない方が使いやすい場合もある
        elif not valid_seed_nodes: # 有効なシードがない (入力が空だった場合も含む)
             st.sidebar.warning("有効なシードノードが選択/入力されていません。")

    except ValueError:
        st.sidebar.error("シードノードはカンマ区切りの数値で入力してください。")
        st.stop()
else: # 入力が空の場合
    st.sidebar.warning("シードノードが入力されていません。")


propagation_prob = st.sidebar.slider("伝播確率 (p)", 0.01, 1.0, 0.1, 0.01, key="pl_prob")

if st.sidebar.button("伝播シミュレーション実行", key="pl_run_sim_button"):
    if not valid_seed_nodes:
        st.sidebar.warning("有効なシードノードがありません。再度確認してください。")
    else:
        activated_n_set, prop_log_list = simulate_ic(G, valid_seed_nodes, propagation_prob)
        st.session_state['pl_activated_nodes_final'] = activated_n_set
        st.session_state['pl_propagation_log_raw'] = prop_log_list # 元のログも保存
        st.session_state['pl_sim_seeds'] = valid_seed_nodes # シミュレーション時のシードも保存
        st.sidebar.success(f"{len(st.session_state['pl_activated_nodes_final'])} ノードが最終的に活性化しました。")

        # ステップごとの「そのステップ終了時点での」全活性化ノードリストを作成
        stepwise_cumulative_activated_nodes = {0: set(valid_seed_nodes)} # ステップ0は初期シード
        current_cumulative_activated = set(valid_seed_nodes)
        if prop_log_list: # ログがある場合のみ処理
            log_df_for_steps = pd.DataFrame(prop_log_list)
            max_step_calc = int(log_df_for_steps['step'].max()) if not log_df_for_steps.empty else 0

            for step in range(1, max_step_calc + 1):
                newly_activated_this_step = set(log_df_for_steps[log_df_for_steps['step'] == step]['target'].unique())
                current_cumulative_activated.update(newly_activated_this_step)
                stepwise_cumulative_activated_nodes[step] = current_cumulative_activated.copy()
        st.session_state['pl_stepwise_cumulative_activated_nodes'] = stepwise_cumulative_activated_nodes


# --- ログと可視化の表示 ---
if 'pl_propagation_log_raw' in st.session_state:
    st.subheader("影響伝播ログ")
    raw_log_df = pd.DataFrame(st.session_state.get('pl_propagation_log_raw', []))
    initial_seeds_for_log = st.session_state.get('pl_sim_seeds', [])


    if not raw_log_df.empty or initial_seeds_for_log: # 初期シードだけでも表示
        # ステップごとにどのノードが「新たに」活性化されたかを表示
        st.write("ステップごとに新たに活性化されたノード:")
        stepwise_newly_activated_display = {}
        # ステップ0 (初期シード)
        stepwise_newly_activated_display[0] = sorted(list(set(initial_seeds_for_log))) # 初期シードをセットとして扱う
        all_nodes_activated_so_far = set(initial_seeds_for_log) # これまでに活性化した全ノード

        max_step_display = 0
        if not raw_log_df.empty:
            max_step_display = int(raw_log_df['step'].max())

        for step_num in range(1, max_step_display + 1):
            nodes_in_log_this_step = set(raw_log_df[raw_log_df['step'] == step_num]['target'].unique())
            truly_new_this_step = nodes_in_log_this_step - all_nodes_activated_so_far
            if truly_new_this_step:
                stepwise_newly_activated_display[step_num] = sorted(list(truly_new_this_step))
            all_nodes_activated_so_far.update(nodes_in_log_this_step)

        # データフレームで見やすく表示
        max_len_nodes = 0
        if stepwise_newly_activated_display:
            max_len_nodes = max(len(nodes) for nodes in stepwise_newly_activated_display.values() if nodes) # 空リストを除外

        display_data_for_df = {}
        for step, nodes in stepwise_newly_activated_display.items():
            if nodes: # ノードリストが空でない場合のみ
                 padded_nodes = nodes + [""] * (max_len_nodes - len(nodes)) # "" でパディング
                 display_data_for_df[f"Step {step}"] = padded_nodes
            # else: # 新規活性化がなかったステップは表示しないか、"-" などで埋めるか
            #     display_data_for_df[f"Step {step}"] = ["-"] * max_len_nodes


        if display_data_for_df:
            newly_activated_df = pd.DataFrame(display_data_for_df)
            st.dataframe(newly_activated_df, height=200, use_container_width=True)
        elif initial_seeds_for_log : # 初期シードのみで伝播なしの場合
            st.write(pd.DataFrame({ "Step 0": sorted(list(initial_seeds_for_log))}))
        else:
            st.write("ログデータがありません。")


        st.write(f"最終的に活性化したノード ({len(st.session_state.get('pl_activated_nodes_final', []))}個):")
        st.write(sorted(list(st.session_state.get('pl_activated_nodes_final', []))))

        # --- ステップごとのグラフ可視化 ---
        st.subheader("ステップごとのグラフ状態可視化")
        stepwise_cumulative_map = st.session_state.get('pl_stepwise_cumulative_activated_nodes', {})
        initial_seeds_for_log = st.session_state.get('pl_sim_seeds', []) # initial_seeds_for_log をここで取得

        if stepwise_cumulative_map: # ステップごとの累積活性化ノード情報があるか
            max_slider_step = max(stepwise_cumulative_map.keys()) if stepwise_cumulative_map else 0

            if max_slider_step > 0: # max_slider_stepが0より大きい場合のみスライダーを表示
                selected_step_slider = st.slider(
                    "表示するステップを選択:",
                    min_value=0, # ステップ0 (初期シード) から
                    max_value=max_slider_step,
                    value=max_slider_step, # デフォルトは最終ステップ
                    key="pl_step_slider_main"
                )
            elif max_slider_step == 0 and 0 in stepwise_cumulative_map: # ステップ0のみ存在する場合
                st.write("ステップ 0 (初期シード状態) のみ表示します。伝播はありませんでした。")
                selected_step_slider = 0 # 表示ステップを0に固定
            else: # stepwise_cumulative_mapが空など、予期せぬ場合
                st.info("表示できる伝播ステップがありません。")
                st.stop() # これ以上処理を進めない

            # 選択されたステップまでの全活性化ノード
            nodes_active_at_selected_step = stepwise_cumulative_map.get(selected_step_slider, set())
            # 選択されたステップで「新たに」活性化されたノード
            newly_activated_at_selected_step = set()
            if selected_step_slider > 0: # selected_step_sliderが定義されている場合のみ
                nodes_active_at_prev_step = stepwise_cumulative_map.get(selected_step_slider - 1, set())
                newly_activated_at_selected_step = nodes_active_at_selected_step - nodes_active_at_prev_step


            nodes_vis_list = []
            # initial_seeds_for_log が未定義の場合のフォールバック
            current_initial_seeds = initial_seeds_for_log if initial_seeds_for_log is not None else set()

            for node_id in G.nodes():
                color = "#E0E0E0" # 非活性 (薄いグレー)
                size = 12
                shape = "dot"
                border_width = 0
                # border_color = "black" # デフォルト (未使用なのでコメントアウト)

                if node_id in nodes_active_at_selected_step:
                    if node_id in current_initial_seeds: # current_initial_seeds を使用
                        color = "red" # 初期シード
                        size = 25
                        shape = "star"
                    elif node_id in newly_activated_at_selected_step:
                        color = "orange" # このステップで新規活性化
                        size = 20
                        border_width = 2 # 新規活性を強調
                        # border_color = "black" # (未使用なのでコメントアウト)
                    else:
                        color = "#FFD700" # 既に活性化済み (Goldなど、オレンジより明るい色)
                        size = 18
                nodes_vis_list.append(Node(id=str(node_id), label=str(node_id), color=color, size=size, shape=shape,
                                       borderWidth=border_width, borderWidthSelected=border_width+1,
                                       labelHighlightBold=True,
                                       font={'color': 'black', 'size': 10 if size <15 else 12}))

            edges_vis_list = []
            # エッジの色分け
            # raw_log_df が未定義の場合のフォールバック
            current_raw_log_df = raw_log_df if 'raw_log_df' in locals() and raw_log_df is not None else pd.DataFrame()

            for u, v in G.edges():
                edge_color = "#E0E0E0" # デフォルト (非活性エッジ)
                edge_width = 1
                is_active_edge_in_log = False
                if not current_raw_log_df.empty: # current_raw_log_df を使用
                    if not current_raw_log_df[(current_raw_log_df['source']==u) & (current_raw_log_df['target']==v) & (current_raw_log_df['step'] <= selected_step_slider)].empty:
                        is_active_edge_in_log = True

                if is_active_edge_in_log and u in nodes_active_at_selected_step and v in nodes_active_at_selected_step:
                    edge_color = "blue"
                    edge_width = 2.5
                elif u in nodes_active_at_selected_step and v in nodes_active_at_selected_step :
                    edge_color = "#B0C4DE"

                edges_vis_list.append(Edge(source=str(u), target=str(v), color=edge_color, width=edge_width, smooth=False))


            config = Config(width="100%", height=700, directed=G.is_directed(),
                            physics={'enabled': True, 'solver': 'forceAtlas2Based',
                                     'forceAtlas2Based': {'gravitationalConstant': -30, 'springLength': 100}},
                            interaction={'hover': True, 'tooltipDelay': 200},
                            nodes={'font': {'size': 10}},
                            edges={'smooth': {'type': 'continuous'}})


            if nodes_vis_list:
                st.write(f"**ステップ {selected_step_slider} の状態:** (赤: 初期シード, オレンジ (太枠): このステップで新規活性化, 金色: 既に活性化, グレー: 未活性)")
                agraph(nodes=nodes_vis_list, edges=edges_vis_list, config=config)
            else:
                st.write("表示するノードがありません。")
        else: # stepwise_cumulative_map が空の場合 (シミュレーション未実行など)
            st.info("シミュレーションを実行すると、ステップごとの可視化が表示されます。")

    # ... (以降のコードは同じ) ...
    elif 'pl_sim_seeds' in st.session_state : # シミュレーションボタンは押したがログがない場合（伝播しなかった場合）
        st.info("シードノードからの新たな活性化はありませんでした。")
        st.write(f"初期シード: {sorted(list(st.session_state.get('pl_sim_seeds', [])))}")

    else:
        st.info("サイドバーで「伝播シミュレーション実行」ボタンを押してログを生成してください。")

else:
    st.info("先に「グラフ可視化」タブでグラフを生成し、その後、このページでシミュレーションを実行してください。")