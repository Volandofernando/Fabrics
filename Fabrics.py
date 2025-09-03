import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils import load_config, load_datasets, detect_features_and_target, train_model, evaluate_model

# -------------------------------
# Load Config & Page Setup
# -------------------------------
config = load_config()

st.set_page_config(
    page_title=f"{config['app']['title']} | UWL Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(f"""
<style>
    .main {{ background-color: #F9FAFB; }}
    h1, h2, h3, h4 {{
        color: {config['app']['theme_color']};
        font-family: 'Helvetica Neue', sans-serif;
    }}
    .stMetric label {{
        font-size: 14px;
        font-weight: 600;
        color: #374151;
    }}
    .recommend-card {{
        padding: 1rem;
        background-color: #FFFFFF;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# App Header
# -------------------------------
st.markdown("<h1 style='text-align: center;'>👕 Fabric Comfort Recommender</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; font-size:18px; color:#4B5563;'>{config['app']['subtitle']}</p>", unsafe_allow_html=True)

st.markdown("---")

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
# Sidebar Navigation
# -------------------------------
st.sidebar.image("logo.png", use_column_width=True)  # optional branding logo
page = st.sidebar.radio("📌 Navigation", ["Get Recommendations", "Dataset Insights", "Model Performance", "About Project"])

# -------------------------------
# PAGE 1: Recommendations
# -------------------------------
if page == "Get Recommendations":
    st.header("📌 Personalized Fabric Recommendation")

    with st.sidebar:
        st.subheader("Set Your Conditions")
        temperature = st.slider("🌡️ Temperature (°C)", 15, 40, 30)
        humidity = st.slider("💧 Humidity (%)", 10, 100, 70)
        sweat_sensitivity = st.radio("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
        activity_intensity = st.radio("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map, activity_map = {"Low": 1, "Medium": 2, "High": 3}, {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    user_input = np.array([[sweat_num * 5,
                            800 + humidity * 5,
                            60 + activity_num * 10,
                            0.04 + (temperature - 25) * 0.001]])
    user_input_scaled = scaler.transform(user_input)

    predicted_score = model.predict(user_input_scaled)[0]
    df_clean["predicted_diff"] = abs(df_clean[target_col] - predicted_score)
    top_matches = df_clean.sort_values(by="predicted_diff").head(3)

    st.markdown("### 🔹 Recommended Fabrics")
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_matches.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div class="recommend-card">
                <h3>🧵 {row.get('fabric_type','Unknown')}</h3>
                <p style="font-size:16px; color:#111827;">Comfort Score</p>
                <h2 style="color:{config['app']['theme_color']};">{round(row[target_col], 2)}</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📊 Comfort Score Comparison")
    chart_data = top_matches[[target_col, "fabric_type"]].rename(columns={target_col: "Comfort Score"})
    chart = alt.Chart(chart_data).mark_bar(color=config["app"]["theme_color"]).encode(
        x=alt.X("fabric_type", sort=None, title="Fabric Type"),
        y=alt.Y("Comfort Score", title="Comfort Score")
    )
    st.altair_chart(chart, use_container_width=True)

    st.download_button("📥 Download Recommendations", top_matches.to_csv(index=False).encode("utf-8"), "fabric_recommendations.csv")

# -------------------------------
# PAGE 2: Dataset Insights
# -------------------------------
elif page == "Dataset Insights":
    st.header("📊 Dataset Insights")
    st.dataframe(df_clean.head(20))

# -------------------------------
# PAGE 3: Model Performance
# -------------------------------
elif page == "Model Performance":
    st.header("🤖 Model Performance")
    metrics = evaluate_model(model, X_test, y_test)
    st.metric("R² Score", metrics["r2"])
    st.metric("RMSE", metrics["rmse"])

# -------------------------------
# PAGE 4: About Project
# -------------------------------
else:
    st.header("📝 About This Project")
    st.markdown(f"""
    **{config['app']['title']}**  
    Developed for the **University of West London** as part of a BSc Dissertation.  

    🚀 Key Features:  
    - AI-powered comfort prediction for fabrics  
    - Combines academic + survey datasets  
    - Professional dashboard-style interface  
    - Supports sportswear, performance clothing, and fashion research  

    👨‍💻 Author: *Volando Fernando*  
    """)

