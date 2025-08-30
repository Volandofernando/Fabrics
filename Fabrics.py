import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model

# -------------------------------
# Load Config
# -------------------------------
config = load_config()

st.set_page_config(page_title=config["app"]["title"], layout="wide")

st.markdown(f"""
<style>
    .main {{ background-color: #F9FAFB; }}
    h1, h2, h3 {{ color: {config['app']['theme_color']}; }}
</style>
""", unsafe_allow_html=True)

st.title(f"👕 {config['app']['title']}")
st.subheader(config["app"]["subtitle"])

# -------------------------------
# Load Data
# -------------------------------
df = load_datasets(config)
feature_cols, target_col = detect_features_and_target(df, config)

if target_col is None or len(feature_cols) < 4:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()

model, scaler, X_test, y_test, df_clean = train_model(df, feature_cols, target_col, config)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Recommendation", "📊 Dataset Insights", "🤖 Model Performance", "📝 About Project"])

# -------------------------------
# TAB 1: Recommendation
# -------------------------------
with tab1:
    st.sidebar.header("Input Your Conditions")
    temperature = st.sidebar.slider("🌡️ Temperature (°C)", 15, 40, 30)
    humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 70)
    sweat_sensitivity = st.sidebar.selectbox("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
    activity_intensity = st.sidebar.selectbox("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map, activity_map = {"Low": 1, "Medium": 2, "High": 3}, {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    user_input = np.array([[
        sweat_num * 5,
        800 + humidity * 5,
        60 + activity_num * 10,
        0.04 + (temperature - 25) * 0.001
    ]])
    user_input_scaled = scaler.transform(user_input)

    predicted_score = model.predict(user_input_scaled)[0]
    df_clean["predicted_diff"] = abs(df_clean[target_col] - predicted_score)
    top_matches = df_clean.sort_values(by="predicted_diff").head(3)

    st.markdown("## 🔹 Top Fabric Recommendations")
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_matches.iterrows()):
        with cols[i]:
            st.markdown(f"### 🧵 {row.get('fabric_type','Unknown')}")
            st.metric("Comfort Score", round(row[target_col], 2))

    chart_data = top_matches[[target_col, "fabric_type"]].rename(columns={target_col:"Comfort Score"})
    chart = alt.Chart(chart_data).mark_bar().encode(x="fabric_type", y="Comfort Score")
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# TAB 2: Dataset Insights
# -------------------------------
with tab2:
    st.markdown("### 📊 Explore Dataset")
    st.dataframe(df_clean.head(10))

# -------------------------------
# TAB 3: Model Performance
# -------------------------------
with tab3:
    metrics = evaluate_model(model, X_test, y_test)
    st.metric("R² Score", metrics["r2"])
    st.metric("RMSE", metrics["rmse"])

# -------------------------------
# TAB 4: About Project
# -------------------------------
with tab4:
    st.markdown(f"""
    **{config['app']['title']}**  
    Developed as part of a BSc Dissertation at the **University of West London**.  

    - Combines literature + real-time survey datasets  
    - Uses Machine Learning (Random Forest) for comfort prediction  
    - Provides professional UI for real-world usability  
    - Includes evaluation metrics and dataset insights  

    📌 Author: *Volando Fernando*  
    """)
