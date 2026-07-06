import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import base64
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_iris

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Iris AI Dashboard",
    page_icon="🌸",
    layout="wide"
)

# ================= CSS & TYPOGRAPHY =================
def load_css(bg_image_file):
    encoded = ""
    if os.path.exists(bg_image_file):
        with open(bg_image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
    
    st.markdown(f"""
    <style>
    /* 1. Google Fonts: Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif !important;
    }}

    /* 2. Background and App Container Setup */
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stAppViewBlockContainer"] {{
        background: rgba(0, 0, 0, 0.50);
        padding: 3rem !important;
        border-radius: 25px;
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-top: 3rem;
        margin-bottom: 3rem;
        max-width: 950px;
    }}

    /* 3. Typography Overrides */
    h1, h2, h3, h4, p, label, .stMarkdown, span {{
        color: #f3e8ff !important; /* Bright soft purple for general text */
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 10px 10px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #e9d5ff !important;
        transition: 0.3s;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(255, 255, 255, 0.2);
        font-weight: 600;
    }}

    /* 4. Input Fields */
    div[data-baseweb="input"] {{
        background: rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }}
    div[data-baseweb="input"]:hover {{
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255, 105, 180, 0.5);
        transform: translateY(-2px);
    }}
    input {{
        color: #ffffff !important;
        font-weight: 500;
    }}

    /* 5. Button */
    .stButton>button {{
        width: 100%;
        height: 55px;
        border-radius: 15px;
        background: linear-gradient(135deg, rgba(255,105,180,0.7) 0%, rgba(148,0,211,0.7) 100%);
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.4);
        transition: all 0.4s ease;
        font-weight: 600;
        font-size: 16px;
        letter-spacing: 1px;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, rgba(255,105,180,0.9) 0%, rgba(148,0,211,0.9) 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 20px rgba(255, 105, 180, 0.4);
    }}

    /* 6. Animations */
    @keyframes float {{
        0% {{ transform: translateY(0px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
        50% {{ transform: translateY(-10px); box-shadow: 0 15px 25px rgba(255,105,180,0.3); }}
        100% {{ transform: translateY(0px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
    }}
    .floating-card {{
        animation: float 4s ease-in-out infinite;
        background: rgba(20, 20, 20, 0.7);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        border: 1px solid rgba(255,105,180,0.5);
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    .result-text {{
        color: #ff4fa3;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0px 0px 10px rgba(255, 79, 163, 0.4);
    }}
    </style>
    """, unsafe_allow_html=True)

load_css("bg.jpg")

# ================= DATA SETUP & AUTO-TRAIN =================
@st.cache_data
def load_real_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    df['Species'] = [iris.target_names[i] for i in iris.target]
    return df, iris.target_names

df_iris, species_names = load_real_data()

if not os.path.exists("iris_model.pkl") or not os.path.exists("scaler.pkl"):
    st.info("🔄 First run detected: Training model and building 4-feature scaler...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    X = df_iris[['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']].values
    y = load_iris().target
    
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X)
    
    model_temp = RandomForestClassifier(random_state=42)
    model_temp.fit(X_scaled, y)
    
    joblib.dump(scaler_temp, "scaler.pkl")
    joblib.dump(model_temp, "iris_model.pkl")
    st.success("✅ Model and scaler created successfully! Please refresh the page.")
    st.stop()

model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

# ================= UI HEADER =================
# INLINE BRINJAL STYLE (Un-ignorable by the browser)
brinjal_style = "color: #4A0E4E !important; font-weight: 700 !important; text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.95), 0px 0px 3px rgba(255, 255, 255, 0.6) !important;"

st.markdown(f"<h1 style='{brinjal_style} text-align: center; font-size: 2.8rem; margin-bottom: 0.5rem;'>🌸 Iris AI Classification Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='{brinjal_style} text-align: center; font-size: 1.2rem; margin-bottom: 2rem;'>Intelligent botanical classification through an elegant, real-time glass interface</p>", unsafe_allow_html=True)

# ================= TABS NAVIGATION =================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction Mode", "📊 Real Dataset Viz", "🎯 Confusion Matrix"])

# ================= TAB 1: PREDICTION =================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"<h3 style='{brinjal_style} text-align: center;'>🌿 Sepal Features</h3>", unsafe_allow_html=True)
        sepal_length = st.number_input("Length (cm)", value=5.8, step=0.1, key="sl")
        sepal_width  = st.number_input("Width (cm)", value=3.0, step=0.1, key="sw")

    with col2:
        st.markdown(f"<h3 style='{brinjal_style} text-align: center;'>🌸 Petal Features</h3>", unsafe_allow_html=True)
        petal_length = st.number_input("Length (cm)", value=4.0, step=0.1, key="pl")
        petal_width  = st.number_input("Width (cm)", value=1.2, step=0.1, key="pw")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Predict Species"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.markdown(f"""
        <div class="floating-card">
            <h3 style="{brinjal_style} margin-bottom: 5px;">Prediction Result</h3>
            <p class="result-text">{prediction}</p>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_scaled)[0]
            classes = model.classes_

            st.markdown(f"<h3 style='{brinjal_style} text-align: center; margin-top: 1.5rem;'>📊 Confidence Rings</h3>", unsafe_allow_html=True)
            
            df_polar = pd.DataFrame({"Confidence": prob * 100, "Class": classes})
            
            fig_polar = px.bar_polar(
                df_polar, r="Confidence", theta="Class",
                color="Class", 
                color_discrete_sequence=["#00FFFF", "#007AFF", "#001F3F"],
                template="plotly_dark"
            )
            
            fig_polar.update_traces(
                marker=dict(line=dict(color="#000000", width=2))
            )
            
            fig_polar.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 100], showticklabels=False, visible=False),
                    angularaxis=dict(direction="clockwise", rotation=90)
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3e8ff", size=14),
                height=380,
                margin=dict(t=30, b=30, l=20, r=20)
            )
            
            st.plotly_chart(fig_polar, use_container_width=True, config={'displayModeBar': False})

# ================= TAB 2: DATASET VISUALIZATION =================
with tab2:
    st.markdown(f"<h3 style='{brinjal_style} margin-bottom: 1rem;'>📈 Sepal vs Petal Distribution</h3>", unsafe_allow_html=True)
    
    fig_scatter = px.scatter(
        df_iris, x="Sepal Length", y="Sepal Width", color="Species", 
        size="Petal Length", hover_data=["Petal Width"],
        color_discrete_sequence=["#ff4fa3", "#00e5ff", "#b558ff"],
        template="plotly_dark"
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3e8ff"),
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ================= TAB 3: CONFUSION MATRIX =================
with tab3:
    st.markdown(f"<h3 style='{brinjal_style}'>🎯 Model Accuracy Assessment</h3>", unsafe_allow_html=True)
    st.write("Simulated model performance mapping (Actual vs Predicted classifications).")
    
    z_matrix = [[49, 1, 0], [0, 48, 2], [0, 3, 47]]
    
    fig_cm = px.imshow(
        z_matrix, text_auto=True, 
        x=species_names, y=species_names,
        labels=dict(x="Predicted Species", y="Actual Species", color="Count"),
        color_continuous_scale="RdPu",
        template="plotly_dark"
    )
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3e8ff"),
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_cm, use_container_width=True)
