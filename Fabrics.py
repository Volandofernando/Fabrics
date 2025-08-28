# ================================
# 📌 Professional Fabric Recommender Web App (Dissertation-Ready)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import altair as alt
import shap

# -------------------------------
# PAGE CONFIG & HEADER
# -------------------------------
st.set_page_config(page_title="AI Fabric Recommender", layout="wide")

st.markdown("""
<style>
    .main { background-color: #F9FAFB; }
    h1, h2, h3 { color: #1F4E79; }
    .stMetric label { font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("👕 AI-Powered Fabric Recommendation System")
st.subheader("Dissertation Project — University of West London")

# -------------------------------
# STEP 1: Load & Clean Data
# -------------------------------
@st.cache_data
def load_data():
    url_lit = "https://raw.githubusercontent.com/Volandofernando/Material-Literature-data-/main/Dataset.xlsx"
    url_survey = "https://raw.githubusercontent.com/Volandofernando/REAL-TIME-Dataset/main/IT%20Innovation%20in%20Fabric%20Industry%20%20(Responses).xlsx"
    df_lit = pd.read_excel(url_lit)
    df_survey = pd.read_excel(url_survey)

    def clean_columns(df):
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]", "_", regex=True)
        return df

    return pd.concat([clean_columns(df_lit), clean_columns(df_survey)], ignore_index=True, sort=False)

df = load_data()

# -------------------------------
# STEP 2: Detect Features & Target
# -------------------------------
feature_keywords = {
    "moisture_regain": ["moisture", "regain"],
    "water_absorption": ["water", "absorption"],
    "drying_time": ["drying", "time"],
    "thermal_conductivity": ["thermal", "conductivity"]
}
target_keywords = ["comfort", "score"]

def find_column(df_cols, keywords):
    for col in df_cols:
        if all(k in col for k in keywords):
            return col
    return None

feature_cols = [find_column(df.columns, kw) for kw in feature_keywords.values()]
target_col = find_column(df.columns, target_keywords)
feature_cols = [c for c in feature_cols if c is not None]

if target_col is None or len(feature_cols) < 4:
    st.error("❌ Dataset error: required features/target not found!")
    st.stop()

# -------------------------------
# STEP 3: Train Model
# -------------------------------
df_clean = df.dropna(subset=feature_cols + [target_col])
X, y = df_clean[feature_cols], df_clean[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# TABS for Navigation
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Recommendation", "📊 Dataset Insights", "🤖 Model Performance", "📝 About Project"])

# -------------------------------
# TAB 1: RECOMMENDATION
# -------------------------------
with tab1:
    st.sidebar.header("Input Your Conditions")
    temperature = st.sidebar.slider("🌡️ Temperature (°C)", 15, 40, 30)
    humidity = st.sidebar.slider("💧 Humidity (%)", 10, 100, 70)
    sweat_sensitivity = st.sidebar.selectbox("🧍 Sweat Sensitivity", ["Low", "Medium", "High"])
    activity_intensity = st.sidebar.selectbox("🏃 Activity Intensity", ["Low", "Moderate", "High"])

    sweat_map, activity_map = {"Low": 1, "Medium": 2, "High": 3}, {"Low": 1, "Moderate": 2, "High": 3}
    sweat_num, activity_num = sweat_map[sweat_sensitivity], activity_map[activity_intensity]

    # Construct features
    user_input = np.array([[
        sweat_num * 5,
        800 + humidity * 5,
        60 + activity_num * 10,
        0.04 + (temperature - 25) * 0.001
    ]])
    user_input_scaled = scaler.transform(user_input)

    # Prediction
    predicted_score = model.predict(user_input_scaled)[0]
    df_clean["predicted_diff"] = abs(df_clean[target_col] - predicted_score)
    top_matches = df_clean.sort_values(by="predicted_diff").head(3)

    st.markdown("## 🔹 Top Fabric Recommendations")
    cols = st.columns(3)
    for i, (_, row) in enumerate(top_matches.iterrows()):
        with cols[i]:
            st.markdown(f"### 🧵 {row.get('fabric_type','Unknown')}")
            st.metric("Comfort Score", round(row[target_col], 2))
            st.caption(f"Moisture: {row[feature_cols[0]]} | Absorption: {row[feature_cols[1]]} | Drying: {row[feature_cols[2]]} | Thermal: {row[feature_cols[3]]}")

    # Chart
    chart_data = top_matches[[target_col, "fabric_type"]].rename(columns={target_col:"Comfort Score"})
    chart = alt.Chart(chart_data).mark_bar().encode(x="fabric_type", y="Comfort Score", tooltip=["fabric_type","Comfort Score"])
    st.altair_chart(chart, use_container_width=True)

# -------------------------------
# TAB 2: DATASET INSIGHTS
# -------------------------------
with tab2:
    st.markdown("### 📊 Explore Dataset Trends")
    st.write("Correlation heatmap, distributions, or fabric property comparisons can be shown here.")
    st.dataframe(df_clean.head(10))

# -------------------------------
# TAB 3: MODEL PERFORMANCE
# -------------------------------
with tab3:
    from sklearn.metrics import mean_squared_error, r2_score
    preds = model.predict(X_test)
    st.metric("R² Score", round(r2_score(y_test, preds), 3))
    st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, preds)), 3))

    st.write("### 🔎 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
    feat_chart = alt.Chart(feat_df).mark_bar(color="#1F77B4").encode(x="Feature", y="Importance")
    st.altair_chart(feat_chart, use_container_width=True)

# -------------------------------
# TAB 4: ABOUT PROJECT
# -------------------------------
with tab4:
    st.markdown("""
    **AI-Powered Fabric Recommender**  
    Developed as part of a BSc Dissertation at the **University of West London**.  

    - Combines literature + real-time survey datasets  
    - Uses Machine Learning (Random Forest) for comfort prediction  
    - Provides professional UI for real-world usability  
    - Includes evaluation metrics and dataset insights  

    📌 Author: *Your Name*  
    """)

