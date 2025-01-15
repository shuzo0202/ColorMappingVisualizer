import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

def main():
    st.title("Lab空間上の色分類可視化アプリ")

    # 1. CSVアップロード
    uploaded_file = st.file_uploader("学習用CSVファイルをアップロード", type=["csv"])
    if uploaded_file is not None:
        # CSV読み込み
        df = pd.read_csv(uploaded_file)
        st.write("アップロードしたデータ（先頭5行）:")
        st.write(df.head())

        # 2. 機械学習の訓練
        X = df[["L*", "a*", "b*"]]
        y = df["色分類"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        st.write("学習完了！")

        # 3. Lab空間上で格子点（メッシュ）を作り予測
        l_range = st.slider("L* の表示範囲", 0, 100, (0, 100), 1)
        a_range = st.slider("a* の表示範囲", -128, 127, (-50, 50), 1)
        b_range = st.slider("b* の表示範囲", -128, 127, (-50, 50), 1)

        step = st.number_input("メッシュのステップ数 (数値が小さいほど粗い)", 
                               min_value=10, max_value=200, value=30)

        L_values = np.linspace(l_range[0], l_range[1], step)
        A_values = np.linspace(a_range[0], a_range[1], step)
        B_values = np.linspace(b_range[0], b_range[1], step)

        mesh_points = np.array(np.meshgrid(L_values, A_values, B_values)).T.reshape(-1, 3)
        predictions = model.predict(mesh_points)

        plot_df = pd.DataFrame(mesh_points, columns=["L*", "a*", "b*"])
        plot_df["pred"] = predictions

        # 4. 色分類ごとに色を指定した辞書
        color_map = {
            "高白色": "#f0f0ff",
            "白":     "#eeeeee",
            "ナチュラル": "#d2b48c",  # タン色
            "黒":     "#000000",
            "グレー": "#808080",
            "赤":     "#ff0000",
            "オレンジ": "#ffa500",
            "茶色":   "#965042",
            "黄色":   "#ffff00",
            "緑":     "#00ff00",
            "青":     "#0000ff",
            "紫":     "#800080",
            "ピンク": "#ffc0cb",
            "金":     "#ffd700",
            "銀":     "#c0c0c0"
        }

        # 5. 3Dプロットで可視化: 軸を変更し、color_discrete_mapを指定
        fig = px.scatter_3d(
            plot_df,
            x="a*", y="b*", z="L*",  # x軸: a*, y軸: b*, z軸: L*
            color="pred",
            color_discrete_map=color_map,
            marker=dict(
                size=6,
                opacity=0.6
            ),
            title="Lab空間における色分類結果"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("CSVファイルをアップロードしてください。")

if __name__ == "__main__":
    main()
