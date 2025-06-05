"""
streamlit_app.py

Main Streamlit application file for navigation.
Located in the streamlit folder.
"""
import streamlit as st

st.set_page_config(
    page_title="情報拡散シミュレーション",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.success("上のメニューからページを選択してください。")

st.title("情報拡散シミュレーションプラットフォーム")
st.write("""
ようこそ！このプラットフォームでは、以下の機能を利用できます。

- **グラフ可視化**: ランダムなソーシャルネットワークグラフを生成し、その構造を視覚的に確認できます。
- **伝播ログ**: 生成したグラフ上で影響伝播シミュレーションを実行し、その結果（どのノードがいつ活性化したかなど）をログとして確認できます。

サイドバーのナビゲーションから各ページにアクセスしてください。
""")
st.markdown("---")
st.caption("将来的には、様々な影響最大化アルゴリズムの比較や、詳細な分析機能を追加予定です。")