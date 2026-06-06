import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline, griddata

# 1. Page Configuration
st.set_page_config(
    page_title="3D Data Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS styling (Dark Glassmorphic Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: radial-gradient(circle at 50% 50%, #111827 0%, #030712 100%) !important;
        color: #f3f4f6 !important;
    }
    
    .main-title {
        background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
        letter-spacing: -0.05rem;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
        font-weight: 300;
    }

    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] b,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #f3f4f6 !important;
    }
    
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #05070f !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #ffffff !important;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    button, .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.3) !important;
    }
    
    button:hover, .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
        color: #f3f4f6 !important;
        border-radius: 8px !important;
    }
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.01);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }

    button[data-testid="StyledFullScreenButton"] {
        display: none !important;
    }
    
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* Plotlyの操作メニューバー（Modebar）を常に見やすくカスタマイズ */
    .js-plotly-plot .plotly .modebar {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    .js-plotly-plot .plotly .modebar-btn path {
        fill: #a5b4fc !important; /* アイコンをかっこいい発色に */
    }
    .js-plotly-plot .plotly .modebar-btn:hover path {
        fill: #f472b6 !important;
    }
</style>
""", unsafe_allow_html=True)


# 3. CSV Loading & Parsing Helper
def load_csv(uploaded_file):
    try:
        if isinstance(uploaded_file, str):
            with open(uploaded_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
        else:
            first_line = uploaded_file.readline().decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            
        parts = [p.strip() for p in first_line.split(',')]
        
        has_header = False
        for p in parts:
            if not p:
                continue
            try:
                float(p)
            except ValueError:
                has_header = True
                break
                
        if has_header:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None)
                
        df.columns = [str(c).strip() for c in df.columns]
        
        if not has_header:
            default_names = ['X', 'Y', 'Z', 'Value']
            df.columns = default_names[:df.shape[1]]
            
        return df
    except Exception as e:
        st.error(f"CSVデータの読み込み中にエラーが発生しました: {e}")
        return None


# 4. Session State Management
if 'df' not in st.session_state:
    st.session_state.df = None
if 'filename' not in st.session_state:
    st.session_state.filename = None

# Title area
st.markdown('<div class="main-title">3D Data Visualizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">アップロードされたCSVファイルから、インタラクティブな3Dグラフを生成します。</div>', unsafe_allow_html=True)

# 5. Sidebar Layout
st.sidebar.markdown("### 1. データ入力 (CSV)")
uploaded_file = st.sidebar.file_uploader("CSVファイルを選択", type=["csv"], label_visibility="collapsed")

if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if df is not None:
        st.session_state.df = df
        st.session_state.filename = uploaded_file.name
else:
    st.sidebar.markdown("<br><b>またはサンプルデータを選択:</b>", unsafe_allow_html=True)
    col_demo1, col_demo2 = st.sidebar.columns(2)
    with col_demo1:
        if st.button("ローレンツ (3列)", use_container_width=True):
            st.session_state.df = load_csv("lorenz_attractor_3d.csv")
            st.session_state.filename = "lorenz_attractor_3d.csv"
    with col_demo2:
        if st.button("減衰波紋面 (4列)", use_container_width=True):
            st.session_state.df = load_csv("ripple_wave_4d.csv")
            st.session_state.filename = "ripple_wave_4d.csv"

df = st.session_state.df
filename = st.session_state.filename

if df is not None:
    cols_count = df.shape[1]
    
    if cols_count < 3:
        st.error(f"読み込まれたCSVには{cols_count}列しかありません。3Dプロットを行うには、少なくとも3列のデータ（X, Y, Z）が必要です。")
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 2. 表示モード選択")
        
        mode_options = [
            "点群 (3D Scatter)",
            "点と点の間を線で繋ぐ (3D Line)",
            "点と点の間をスプライン曲線で繋ぐ (3D Spline Line)",
            "点群表示＋カラー (Contour Scatter)",
            "サーフェス表示＋コンター (3D Surface Contour)"
        ]
        
        selected_mode = st.sidebar.radio("表示モード", options=mode_options, label_visibility="collapsed")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 3. スタイル調整")
        
        point_size = st.sidebar.slider("点サイズ", min_value=1, max_value=20, value=5, step=1)
        axis_label_size = st.sidebar.slider("軸ラベルサイズ", min_value=10, max_value=30, value=14, step=1)
        
        fig = None
        invalid_mode = False
        
        contour_min, contour_max = 0.0, 1.0
        if cols_count >= 4:
            c_data = df.iloc[:, 3]
            c_min_val = float(c_data.min())
            c_max_val = float(c_data.max())
            
            if c_min_val == c_max_val:
                c_min_val -= 1.0
                c_max_val += 1.0
                
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 4. コンターレンジ (4列目用)")
            contour_range = st.sidebar.slider(
                "レンジ調整 (最小 / 最大)", min_value=c_min_val, max_value=c_max_val, value=(c_min_val, c_max_val),
                step=0.01 if (c_max_val - c_min_val) < 20 else 0.1
            )
            contour_min, contour_max = contour_range
            
        is_4col_mode = selected_mode in ["点群表示＋カラー (Contour Scatter)", "サーフェス表示＋コンター (3D Surface Contour)"]
        if is_4col_mode and cols_count < 4:
            st.warning("⚠️ 4列目のデータを含むCSVをアップロードしてください。現在のデータは3列のみです。")
            invalid_mode = True
            
        if not invalid_mode:
            x_label = df.columns[0]
            y_label = df.columns[1]
            z_label = df.columns[2]
            c_label = df.columns[3] if cols_count >= 4 else "Value"
            
            with st.spinner("3Dグラフを描画中..."):
                if selected_mode == "点群 (3D Scatter)":
                    fig = go.Figure(data=[go.Scatter3d(
                        x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers',
                        marker=dict(size=point_size, color='#6366f1', opacity=0.8, line=dict(width=0))
                    )])
                    
                elif selected_mode == "点と点の間を線で繋ぐ (3D Line)":
                    fig = go.Figure(data=[go.Scatter3d(
                        x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='lines+markers',
                        marker=dict(size=point_size/2, color='#3b82f6', opacity=0.8), line=dict(color='#8b5cf6', width=3)
                    )])
                    
                elif selected_mode == "点と点の間をスプライン曲線で繋ぐ (3D Spline Line)":
                    N = len(df)
                    if N < 3:
                        st.warning("⚠️ データ点数が3点未満のため通常の線グラフで描画します。")
                        fig = go.Figure(data=[go.Scatter3d(
                            x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='lines+markers', marker=dict(size=point_size/2, color='#3b82f6'), line=dict(color='#8b5cf6', width=3)
                        )])
                    else:
                        t = np.arange(N)
                        cs_x = CubicSpline(t, df.iloc[:, 0])
                        cs_y = CubicSpline(t, df.iloc[:, 1])
                        cs_z = CubicSpline(t, df.iloc[:, 2])
                        t_new = np.linspace(0, N - 1, num=max(100, N * 8))
                        if cols_count >= 4:
                            cs_c = CubicSpline(t, df.iloc[:, 3])
                            fig = go.Figure(data=[go.Scatter3d(
                                x=cs_x(t_new), y=cs_y(t_new), z=cs_z(t_new), mode='lines',
                                line=dict(color=cs_c(t_new), colorscale='Viridis', width=4, cmin=contour_min, cmax=contour_max,
                                         colorbar=dict(title=dict(text=c_label, font=dict(size=axis_label_size, color="#e2e8f0")), tickfont=dict(size=axis_label_size - 2, color="#94a3b8")))
                            )])
                        else:
                            fig = go.Figure(data=[go.Scatter3d(x=cs_x(t_new), y=cs_y(t_new), z=cs_z(t_new), mode='lines', line=dict(color='#8b5cf6', width=4))])
                            
                elif selected_mode == "点群表示＋カラー (Contour Scatter)":
                    fig = go.Figure(data=[go.Scatter3d(
                        x=df.iloc[:, 0], y=df.iloc[:, 1], z=df.iloc[:, 2], mode='markers',
                        marker=dict(size=point_size, color=df.iloc[:, 3], colorscale='Viridis', cmin=contour_min, cmax=contour_max,
                                    colorbar=dict(title=dict(text=c_label, font=dict(size=axis_label_size, color="#e2e8f0")), tickfont=dict(size=axis_label_size - 2, color="#94a3b8")), opacity=0.8, line=dict(width=0))
                    )])
                    
                elif selected_mode == "サーフェス表示＋コンター (3D Surface Contour)":
                    x_raw, y_raw, z_raw, c_raw = df.iloc[:, 0].values, df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values
                    if (x_raw.max() - x_raw.min()) == 0 or (y_raw.max() - y_raw.min()) == 0:
                        st.error("❌ X/Y座標の範囲が0のため、サーフェスを生成できません。")
                    else:
                        grid_x_1d = np.linspace(x_raw.min(), x_raw.max(), 100)
                        grid_y_1d = np.linspace(y_raw.min(), y_raw.max(), 100)
                        grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
                        grid_z = griddata((x_raw, y_raw), z_raw, (grid_x, grid_y), method='linear')
                        grid_c = griddata((x_raw, y_raw), c_raw, (grid_x, grid_y), method='linear')
                        if np.isnan(grid_z).any():
                            grid_z = np.where(np.isnan(grid_z), griddata((x_raw, y_raw), z_raw, (grid_x, grid_y), method='nearest'), grid_z)
                            grid_c = np.where(np.isnan(grid_c), griddata((x_raw, y_raw), c_raw, (grid_x, grid_y), method='nearest'), grid_c)
                        fig = go.Figure(data=[go.Surface(
                            x=grid_x_1d, y=grid_y_1d, z=grid_z, surfacecolor=grid_c, colorscale='Viridis', cmin=contour_min, cmax=contour_max,
                            colorbar=dict(title=dict(text=c_label, font=dict(size=axis_label_size, color="#e2e8f0")), tickfont=dict(size=axis_label_size - 2, color="#94a3b8"))
                        )])

            if fig is not None:
                fig.update_layout(
                    scene=dict(
                        xaxis=dict(backgroundcolor="rgba(17, 24, 39, 0.5)", gridcolor="rgba(255, 255, 255, 0.08)", showbackground=True, zerolinecolor="rgba(255, 255, 255, 0.15)", title=dict(text=x_label, font=dict(size=axis_label_size, color="#9ca3af")), tickfont=dict(size=axis_label_size - 2, color="#6b7280")),
                        yaxis=dict(backgroundcolor="rgba(17, 24, 39, 0.5)", gridcolor="rgba(255, 255, 255, 0.08)", showbackground=True, zerolinecolor="rgba(255, 255, 255, 0.15)", title=dict(text=y_label, font=dict(size=axis_label_size, color="#9ca3af")), tickfont=dict(size=axis_label_size - 2, color="#6b7280")),
                        zaxis=dict(backgroundcolor="rgba(17, 24, 39, 0.5)", gridcolor="rgba(255, 255, 255, 0.08)", showbackground=True, zerolinecolor="rgba(255, 255, 255, 0.15)", title=dict(text=z_label, font=dict(size=axis_label_size, color="#9ca3af")), tickfont=dict(size=axis_label_size - 2, color="#6b7280")),
                    ),
                    margin=dict(l=0, r=0, b=0, t=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Outfit, sans-serif")
                )
                
                # Render Main Chart
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                
                # 【新戦略の核心】
                # 1. config で 'toImageButtonOptions' を指定し、カメラボタンを押した時の解像度を縦横2倍、高画質（1800x1200）に固定
                # 2. 'displayModeBar': True で常時右上にメニューを表示
                file_base = filename.split('.')[0]
                st.plotly_chart(
                    fig, 
                    use_container_width=True, 
                    theme=None, 
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'3d_plot_{file_base}',
                            'height': 900,
                            'width': 1350,
                            'scale': 2 # 2倍鮮明に出力
                        }
                    }
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.info("💡 **【画像エクスポート方法】** グラフの右上（ホバー時）に表示される **カメラ型アイコン（Download plot as a png）** をクリックしてください。マウスで調整した**現在の視点・拡大率のまま**、100%確実に高画質なPNG画像が保存されます。")
                        
                # 7. Data Preview Table
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("👁️ 元データプレビュー"):
                    st.markdown(f"**ファイル名:** `{filename}` | **行数:** `{len(df)}` | **列数:** `{cols_count}`")
                    st.dataframe(df, use_container_width=True)

else:
    st.markdown('<div class="glass-card" style="text-align: center; padding: 4rem 2rem;">', unsafe_allow_html=True)
    st.markdown("### 📂 データがロードされていません")
    st.markdown("左側のサイドバーから `.csv` 形式の3D点群データをアップロードするか、サンプルデータボタンを押して開始してください。")
    st.markdown("</div>", unsafe_allow_html=True)

# 8. Shutdown Application Button
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ アプリ操作")
if st.sidebar.button("🔴 アプリを終了する", use_container_width=True):
    st.sidebar.warning("アプリケーションを停止しています...")
    st.markdown('<div class="glass-card" style="text-align: center; padding: 4rem 2rem; border: 1px solid #ef4444; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.2);">'
                '<h3 style="color: #ef4444; font-weight: 700;">🔴 アプリケーションは終了しました</h3>'
                '<p style="color: #9ca3af; margin-top: 1rem;">サーバーが正常にシャットダウンされました。このブラウザタブを閉じてターミナルに戻ってください。</p></div>', unsafe_allow_html=True)
    import os
    import signal
    os.kill(os.getpid(), signal.SIGINT)