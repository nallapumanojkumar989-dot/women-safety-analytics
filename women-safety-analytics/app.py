import os
import sys
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from streamlit_folium import st_folium

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_crime_data, get_data_metadata, DATA_ATTRIBUTION
from src.preprocessing import clean_and_preprocess_data, generate_data_quality_report
from src.feature_engineering import compute_derived_features, get_derived_feature_labels
from src.clustering import run_dbscan_clustering, summarize_clusters
from src.anomaly_detection import run_isolation_forest, get_anomaly_summary, ANOMALY_NOTE
from src.danger_score import calculate_danger_scores, get_area_danger_breakdown, DEFAULT_WEIGHTS
from src.confidence_score import calculate_confidence_score, get_confidence_tooltip
from src.explanations import generate_evidence_explanations
from src.model_training import train_and_evaluate_models, FEATURE_COLUMNS
from src.prediction import predict_area_danger_score, load_prediction_model

from utils.visualization import (
    create_danger_score_gauge,
    create_confidence_gauge,
    create_feature_contribution_chart,
    create_crime_category_pie,
    create_area_comparison_chart,
    create_year_trend_chart,
    build_interactive_folium_map,
    create_model_feature_importance_chart,
    create_model_comparison_chart,
    create_actual_vs_predicted_chart
)

# Page configuration
st.set_page_config(
    page_title="Women Safety Analytics – ML Danger Prediction & Risk Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark UI aesthetics
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        margin-bottom: 12px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #38BDF8;
    }
    .kpi-label {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .disclaimer-box {
        background-color: #1E1B4B;
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 13px;
        line-height: 1.5;
        color: #C7D2FE;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 2px solid #38BDF8;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .reason-item {
        background: #1E293B;
        border-left: 3px solid #38BDF8;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Application Title Header
st.title("🛡️ Women Safety Analytics")
st.caption("### Supervised ML Danger Prediction & Urban Safety Risk Detection System")

# Responsible-use Note (Section 25)
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ RESPONSIBLE-USE NOTE & DISCLAIMER:</b><br>
    This platform analyzes historical open-government crime data published by the National Crime Records Bureau (NCRB) and executes a <b>Random Forest Regressor</b> to predict urban area Danger Scores (0–100). 
    A higher score reflects stronger statistical evidence of crime intensity in the training data; it does NOT guarantee that a crime will occur. 
    The ML models evaluate historical structural patterns, not individual persons.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SIDEBAR CONTROLS & FILTERS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Controls & Filters")

# Dataset Selection
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["NCRB District-Wise Dataset (2014-2022)", "NCRB City-Wise Dataset (2021-2023)"],
    help="Primary dataset covers 600+ real district-year records across 35 States/UTs."
)

data_filename = "ncrb_district_crime_women.csv" if "District" in dataset_choice else "ncrb_crime_women_2023.csv"
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", data_filename)

try:
    df_raw, attribution = load_crime_data(data_path)
    meta = get_data_metadata(data_path)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Pipeline Execution
df_clean = clean_and_preprocess_data(df_raw)
data_quality_info = generate_data_quality_report(df_raw, df_clean)
df_features = compute_derived_features(df_clean)

# ML Sidebar Parameters
st.sidebar.subheader("🤖 ML & Clustering Parameters")
eps_val = st.sidebar.slider("DBSCAN Epsilon (eps)", min_value=0.1, max_value=3.0, value=0.6, step=0.1)
min_samples_val = st.sidebar.slider("DBSCAN Min Samples", min_value=1, max_value=10, value=2, step=1)
use_geo_clustering = st.sidebar.checkbox("Use Geographic Lat/Lon Clustering", value=False)

st.sidebar.subheader("⚖️ Danger Score Weights")
w_freq = st.sidebar.slider("Crime Frequency Weight", 0.0, 0.5, DEFAULT_WEIGHTS["crime_frequency"], 0.05)
w_women = st.sidebar.slider("Women-related Crime Weight", 0.0, 0.5, DEFAULT_WEIGHTS["women_crime_ratio"], 0.05)
w_serious = st.sidebar.slider("Serious Crime Weight", 0.0, 0.5, DEFAULT_WEIGHTS["serious_crime_ratio"], 0.05)
w_intensity = st.sidebar.slider("Relative Intensity Weight", 0.0, 0.5, DEFAULT_WEIGHTS["relative_intensity"], 0.05)
w_trend = st.sidebar.slider("Recent Trend Weight", 0.0, 0.5, DEFAULT_WEIGHTS["recent_trend"], 0.05)
w_anomaly = st.sidebar.slider("Anomaly Signal Weight", 0.0, 0.5, DEFAULT_WEIGHTS["anomaly_signal"], 0.05)

custom_weights = {
    "crime_frequency": w_freq,
    "women_crime_ratio": w_women,
    "serious_crime_ratio": w_serious,
    "relative_intensity": w_intensity,
    "recent_trend": w_trend,
    "anomaly_signal": w_anomaly
}

# Run Anomaly Detection (Isolation Forest)
df_anomaly, iso_model = run_isolation_forest(df_features, contamination=0.1)

# Compute Target Danger Scores
df_scored, active_weights = calculate_danger_scores(df_anomaly, custom_weights)

# Compute Risk Assessment Confidence
df_final = calculate_confidence_score(df_scored, data_quality_info)

# Run DBSCAN Clustering
df_final, dbscan_model = run_dbscan_clustering(df_final, eps=eps_val, min_samples=min_samples_val, use_geographic=use_geo_clustering)

# Train Supervised ML Models (or load if cached)
model_obj, scaler_obj, ml_metrics, actual_vs_pred_df = train_and_evaluate_models(df_final)

# Sidebar Filters
st.sidebar.subheader("🔍 Data Filters")
available_states = sorted(df_final["State"].unique())
selected_state = st.sidebar.selectbox("Filter State", ["All States"] + available_states)

if selected_state != "All States":
    available_cities = sorted(df_final[df_final["State"] == selected_state]["City"].unique())
else:
    available_cities = sorted(df_final["City"].unique())
    
selected_year = st.sidebar.selectbox("Select Year", sorted(df_final["Year"].unique(), reverse=True))

# ---------------------------------------------------------
# TOP KPI CARDS
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_records = len(df_final)
total_districts = len(df_final["City"].unique())
anomalies_cnt = (df_final[df_final["Year"] == selected_year]["Anomaly_Status"] == "Anomalous").sum()
model_r2_val = ml_metrics["test_r2"]

with col1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Dataset Records</div><div class="kpi-value">{total_records:,}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Districts / Areas</div><div class="kpi-value">{total_districts}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">Anomalies Detected ({selected_year})</div><div class="kpi-value">{anomalies_cnt}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">ML Model R² Score</div><div class="kpi-value">{model_r2_val:.2f}</div></div>', unsafe_allow_html=True)

st.markdown(f"**Dataset Source:** `{meta['type']}` — *{meta['description']} ({total_records} rows across {len(data_quality_info['available_years'])} years)*")

# ---------------------------------------------------------
# MAIN TABS ARCHITECTURE
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📍 Area Safety Analysis",
    "🤖 Danger Prediction System",
    "🗺️ Interactive Risk Map",
    "📊 Crime Analytics & Trends",
    "📈 Model Performance & Validation",
    "🤖 ML Clustering & Anomalies",
    "📋 Dataset Governance"
])

# ---------------------------------------------------------
# TAB 1: AREA SAFETY ANALYSIS
# ---------------------------------------------------------
with tab1:
    st.subheader("Urban Area Historical Safety Risk Assessment")
    
    sel_city = st.selectbox("Select District / Area for In-Depth Risk Analysis:", available_cities, index=0)
    
    city_data = df_final[(df_final["City"] == sel_city) & (df_final["Year"] == selected_year)]
    if city_data.empty:
        city_data = df_final[df_final["City"] == sel_city].sort_values(by="Year", ascending=False).iloc[:1]
        
    area_row = city_data.iloc[0]
    
    danger_val = area_row["Danger_Score"]
    conf_val = area_row["Confidence_Score"]
    breakdown = get_area_danger_breakdown(area_row, active_weights)
    reasons = generate_evidence_explanations(area_row, df_final)
    
    # Calculate ML Model Prediction for this area
    input_dict = {f: area_row.get(f, 0.0) for f in FEATURE_COLUMNS}
    pred_res = predict_area_danger_score(input_dict)
    
    col_a, col_b = st.columns([1, 1.2])
    
    with col_a:
        st.markdown(f"### **{sel_city}**, {area_row['State']} ({area_row['Year']})")
        st.write(f"**Primary Concern:** {breakdown['primary_concern']}")
        st.write(f"**Top Contributing Factor:** `{breakdown['top_factor']}`")
        
        # Side-by-Side Comparison of Historical vs Predicted Score
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"""
            <div style="background:#1E293B; padding:12px; border-radius:8px; border-left:4px solid #F97316;">
                <span style="font-size:12px; color:#94A3B8;">HISTORICAL DANGER SCORE</span><br>
                <span style="font-size:24px; font-weight:bold; color:#F97316;">{danger_val} / 100</span>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div style="background:#1E293B; padding:12px; border-radius:8px; border-left:4px solid #38BDF8;">
                <span style="font-size:12px; color:#94A3B8;">PREDICTED ML SCORE</span><br>
                <span style="font-size:24px; font-weight:bold; color:#38BDF8;">{pred_res['predicted_danger_score']} / 100</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.plotly_chart(create_danger_score_gauge(danger_val, "Historical Score"), use_container_width=True)
        with g_col2:
            st.plotly_chart(create_confidence_gauge(conf_val, "Confidence"), use_container_width=True)

    with col_b:
        st.markdown("### **Why is this area considered dangerous?**")
        st.caption("Factual 3–5 evidence-based reasons derived directly from historical NCRB metrics:")
        for r in reasons:
            st.markdown(f'<div class="reason-item">📌 {r}</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(create_feature_contribution_chart(breakdown), use_container_width=True)
    with col_d:
        st.plotly_chart(create_crime_category_pie(area_row), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: DANGER PREDICTION SYSTEM (SUPERVISED ML SECTION)
# ---------------------------------------------------------
with tab2:
    st.subheader("🤖 Random Forest Danger Prediction System")
    st.caption("Execute genuine supervised ML model predictions (`model.predict(input_features)`) using trained Random Forest Regressor artifacts.")
    
    pred_city = st.selectbox("Select Area to Predict & Adjust Features:", available_cities, index=0, key="pred_city_select")
    
    p_city_data = df_final[(df_final["City"] == pred_city) & (df_final["Year"] == selected_year)]
    if p_city_data.empty:
        p_city_data = df_final[df_final["City"] == pred_city].sort_values(by="Year", ascending=False).iloc[:1]
    p_row = p_city_data.iloc[0]
    
    st.markdown("#### 1. Input Features Configuration")
    st.caption("Features are automatically populated with historical averages for the selected area. Adjust sliders to test scenario predictions:")
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        in_norm_intensity = st.slider("Normalized Crime Intensity", 0.0, 1.0, float(p_row.get("Normalized_Crime_Intensity", 0.5)), 0.05)
        in_serious_ratio = st.slider("Serious Crime Ratio", 0.0, 1.0, float(p_row.get("Serious_Crime_Ratio", 0.3)), 0.05)
    with f2:
        in_diversity = st.slider("Crime Category Diversity", 0.0, 3.0, float(p_row.get("Crime_Category_Diversity", 1.5)), 0.1)
        in_concentration = st.slider("Crime Concentration Ratio", 0.0, 1.0, float(p_row.get("Crime_Concentration", 0.6)), 0.05)
    with f3:
        in_rel_intensity = st.slider("Relative Crime Intensity", 0.0, 5.0, float(p_row.get("Relative_Crime_Intensity", 1.0)), 0.1)
        in_yoy_trend = st.slider("YoY Change Pct (%)", -50.0, 50.0, float(p_row.get("YoY_Change_Pct", 0.0)), 1.0)
    with f4:
        in_anomaly_score = st.slider("Anomaly Intensity Score", 0.0, 100.0, float(p_row.get("Anomaly_Score", 10.0)), 5.0)
        in_population = st.number_input("Population (in Lakhs)", min_value=1.0, max_value=300.0, value=float(p_row.get("Population_Lakhs", 25.0)), step=1.0)

    input_user_dict = {
        "Normalized_Crime_Intensity": in_norm_intensity,
        "Serious_Crime_Ratio": in_serious_ratio,
        "Crime_Category_Diversity": in_diversity,
        "Crime_Concentration": in_concentration,
        "Relative_Crime_Intensity": in_rel_intensity,
        "YoY_Change_Pct": in_yoy_trend,
        "Anomaly_Score": in_anomaly_score,
        "Population_Lakhs": in_population
    }
    
    # Execute Model Prediction
    ml_prediction_result = predict_area_danger_score(input_user_dict)
    
    st.markdown("---")
    st.markdown("#### 2. ML Prediction Output")
    
    pcol1, pcol2, pcol3 = st.columns([1, 1, 1.2])
    
    with pcol1:
        st.markdown(f"""
        <div class="prediction-box">
            <span style="font-size:13px; color:#94A3B8; uppercase;">PREDICTED DANGER SCORE</span><br>
            <span style="font-size:42px; font-weight:900; color:#38BDF8;">{ml_prediction_result['predicted_danger_score']} <span style="font-size:22px;">/ 100</span></span><br>
            <span style="font-size:12px; color:#64748B;">Generated by Trained Random Forest Regressor</span>
        </div>
        """, unsafe_allow_html=True)

    with pcol2:
        st.markdown(f"""
        <div class="prediction-box">
            <span style="font-size:13px; color:#94A3B8; uppercase;">HISTORICAL OBSERVED SCORE</span><br>
            <span style="font-size:42px; font-weight:900; color:#F97316;">{p_row['Danger_Score']} <span style="font-size:22px;">/ 100</span></span><br>
            <span style="font-size:12px; color:#64748B;">Based on {p_row['Year']} Historical NCRB Data</span>
        </div>
        """, unsafe_allow_html=True)
        
    with pcol3:
        st.plotly_chart(create_confidence_gauge(ml_prediction_result['prediction_confidence'], "Prediction Reliability"), use_container_width=True)
        st.caption(f"💡 *{ml_prediction_result['confidence_explanation']}*")

    st.markdown("#### 3. Why This Prediction? (Model Feature Importances)")
    st.caption("The factors that influenced the Random Forest Regressor's output, derived directly from model feature importances:")
    
    wcol1, wcol2 = st.columns([1, 1])
    with wcol1:
        for factor in ml_prediction_result["why_factors"]:
            st.markdown(f"✓ **{factor['factor_name']}**: `{factor['importance_pct']}%` influence on model prediction")
    with wcol2:
        st.plotly_chart(create_model_feature_importance_chart(ml_metrics["feature_importance_df"]), use_container_width=True)

# ---------------------------------------------------------
# TAB 3: INTERACTIVE RISK MAP
# ---------------------------------------------------------
with tab3:
    st.subheader(f"🗺️ Geographic Safety Risk Map ({selected_year})")
    st.info("控制ℹ️ **Geographic Granularity Notice:** Geographic markers are located at official District / City centroids based on NCRB reported data.")
    
    folium_map = build_interactive_folium_map(df_final, selected_year=selected_year)
    st_folium(folium_map, width=1200, height=520)

# ---------------------------------------------------------
# TAB 4: CRIME ANALYTICS & TRENDS
# ---------------------------------------------------------
with tab4:
    st.subheader("📊 District Crime Distribution & Comparative Analytics")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.plotly_chart(create_area_comparison_chart(df_final, current_year=selected_year), use_container_width=True)
    with col_t2:
        st.plotly_chart(create_year_trend_chart(df_final, sel_city), use_container_width=True)

# ---------------------------------------------------------
# TAB 5: MODEL PERFORMANCE & VALIDATION
# ---------------------------------------------------------
with tab5:
    st.subheader("📈 ML Model Performance, Validation & Comparison")
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Selected Model", ml_metrics["best_model_name"])
    mcol2.metric("Test MAE", f"{ml_metrics['test_mae']:.2f}")
    mcol3.metric("Test RMSE", f"{ml_metrics['test_rmse']:.2f}")
    mcol4.metric("Test R² Score", f"{ml_metrics['test_r2']:.4f}")
    
    if ml_metrics.get("possible_overfitting", False):
        st.warning("⚠️ **Possible Overfitting Detected:** The model performs substantially better on training data than validation data. Predictions should be interpreted with caution.")

    st.markdown("#### 1. Candidate Models Comparison (5-Fold Cross-Validation)")
    st.caption("Models compared on 80% training data using 5-Fold Cross Validation:")
    st.dataframe(ml_metrics["comparison_df"], use_container_width=True)
    st.plotly_chart(create_model_comparison_chart(ml_metrics["comparison_df"]), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 2. Test Set Evaluation: Actual vs Predicted Danger Scores")
    st.plotly_chart(create_actual_vs_predicted_chart(actual_vs_pred_df), use_container_width=True)

# ---------------------------------------------------------
# TAB 6: ML CLUSTERING & ANOMALIES
# ---------------------------------------------------------
with tab6:
    st.subheader("🤖 Machine Learning Analytics: DBSCAN & Isolation Forest")
    
    st.markdown("#### 1. Isolation Forest Anomaly Detection")
    st.caption(f"*{ANOMALY_NOTE}*")
    
    anom_summary = get_anomaly_summary(df_final[df_final["Year"] == selected_year])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Observations", anom_summary["total_observations"])
    m2.metric("Detected Anomalies", anom_summary["detected_anomalies"])
    m3.metric("Anomaly Percentage", f"{anom_summary['anomaly_percentage']}%")
    
    st.markdown("**Anomalous Observations Table:**")
    anom_cols = ["City", "State", "Year", "Total_Crime_Against_Women", "Anomaly_Score", "Anomaly_Status", "Recent_Crime_Trend"]
    st.dataframe(
        anom_summary["anomalies_table"][anom_cols].sort_values(by="Anomaly_Score", ascending=False),
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("#### 2. DBSCAN Clustering Analysis")
    st.write(f"Parameters: `eps = {eps_val}`, `min_samples = {min_samples_val}`")
    
    cluster_df_summary = summarize_clusters(df_final[df_final["Year"] == selected_year])
    st.dataframe(cluster_df_summary, use_container_width=True)

# ---------------------------------------------------------
# TAB 7: DATASET GOVERNANCE & AUDIT
# ---------------------------------------------------------
with tab7:
    st.subheader("📋 Dataset Audit & Governance Panel")
    
    dq_col1, dq_col2 = st.columns(2)
    with dq_col1:
        st.markdown("#### **Quality & Coverage Metrics**")
        st.write(f"- **Total Dataset Records:** `{data_quality_info['total_rows']}`")
        st.write(f"- **Total Dataset Columns:** `{data_quality_info['total_columns']}`")
        st.write(f"- **Missing Value Count:** `{data_quality_info['raw_missing_values']}`")
        st.write(f"- **Duplicate Rows Removed:** `{data_quality_info['duplicate_rows_removed']}`")
        st.write(f"- **Data Completeness Score:** `{data_quality_info['data_completeness_pct']}%`")
        st.write(f"- **Available Historical Years:** `{data_quality_info['available_years']}`")
        st.write(f"- **Districts / Cities Count:** `{data_quality_info['total_cities']}`")
        st.write(f"- **States / UTs Count:** `{data_quality_info['total_states']}`")
    
    with dq_col2:
        st.markdown("#### **Official Data Source Attribution**")
        st.write(f"**Publisher:** {DATA_ATTRIBUTION['source_name']}")
        st.write(f"**Catalog:** {DATA_ATTRIBUTION['catalog_title']}")
        st.write(f"**Portal:** [data.gov.in Catalog]({DATA_ATTRIBUTION['official_url']})")
        st.write(f"**Open Data Resource:** [OpenCity Dataset]({DATA_ATTRIBUTION['resource_url']})")
        st.write(f"**Disclaimer:** {DATA_ATTRIBUTION['disclaimer']}")

    st.markdown("---")
    st.markdown("#### **Raw Processed Dataset Viewer**")
    st.dataframe(df_final, use_container_width=True)
