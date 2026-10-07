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
from src.prediction import predict_area_danger_score, load_prediction_model, predict_danger_for_user_input

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
    create_actual_vs_predicted_chart,
    create_risk_trend_chart
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
    "🔮 Future Safety Risk Prediction",
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
# TAB 2: FUTURE SAFETY RISK PREDICTION
# ---------------------------------------------------------
with tab2:
    st.subheader("🔮 ML-Powered Future Safety Risk Prediction")
    st.caption("Forecast future Danger Scores based on historical trends and time-aware ML modeling.")
    
    col_t2a, col_t2b, col_t2c = st.columns(3)
    with col_t2a:
        pred_state = st.selectbox("1. Filter State:", ["All States"] + sorted(df_final["State"].unique()), key="pred_state")
    with col_t2b:
        avail_pred_cities = sorted(df_final[df_final["State"] == pred_state]["City"].unique()) if pred_state != "All States" else sorted(df_final["City"].unique())
        pred_city = st.selectbox("2. Select District / Area:", avail_pred_cities, key="pred_city")
    with col_t2c:
        max_year = int(df_final["Year"].max())
        pred_year = st.selectbox("3. Prediction Year:", list(range(max_year + 1, max_year + 16)), key="pred_target_year")

    if st.button("Predict Future Risk", type="primary"):
        res = predict_danger_for_user_input(df_final, pred_city, pred_year)
        if res["status"] == "error":
            st.error(res["message"])
        else:
            st.markdown("---")
            score = res["predicted_danger_score"]
            hist_score = res["historical_danger_score"]
            diff = score - hist_score
            diff_pct = (diff / max(1.0, hist_score)) * 100
            
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            with pcol1:
                st.markdown(f'<div class="prediction-box"><div style="color:#94A3B8; font-size:12px;">PREDICTED RISK ({pred_year})</div><div style="font-size:36px; font-weight:bold; color:#38BDF8;">{score}/100</div></div>', unsafe_allow_html=True)
            with pcol2:
                st.markdown(f'<div class="prediction-box"><div style="color:#94A3B8; font-size:12px;">LATEST HISTORICAL ({res["historical_year"]})</div><div style="font-size:36px; font-weight:bold; color:#F97316;">{hist_score}/100</div></div>', unsafe_allow_html=True)
            with pcol3:
                arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
                color = "#EF4444" if diff > 5 else ("#22C55E" if diff < -5 else "#EAB308")
                trend_text = "Increasing Risk" if diff > 5 else ("Decreasing Risk" if diff < -5 else "Stable Risk")
                st.markdown(f'<div class="prediction-box"><div style="color:#94A3B8; font-size:12px;">PREDICTED CHANGE</div><div style="font-size:28px; font-weight:bold; color:{color};">{arrow} {abs(diff_pct):.1f}%</div><div style="font-size:14px; color:{color};">{trend_text}</div></div>', unsafe_allow_html=True)
            with pcol4:
                 st.plotly_chart(create_confidence_gauge(res['prediction_confidence'], "Reliability"), use_container_width=True)

            st.write(f"💡 **Reliability Note:** {res['confidence_explanation']}")
            
            # Risk Trend Chart
            st.markdown("#### Historical vs Forecasted Safety Risk")
            city_hist = df_final[df_final["City"] == pred_city].sort_values(by="Year")
            hist_years = city_hist["Year"].tolist()
            hist_scores = city_hist["Danger_Score"].tolist()
            
            # Simulate forecast line
            forecast_years = list(range(max_year + 1, pred_year + 1))
            forecast_scores = []
            multi_year_table = []
            for y in forecast_years:
                f_res = predict_danger_for_user_input(df_final, pred_city, y)
                f_score = f_res.get("predicted_danger_score", score)
                forecast_scores.append(f_score)
                multi_year_table.append({
                    "Year": y,
                    "Predicted Danger Score": f_score,
                    "Risk Level": f_res.get("risk_level", "-"),
                    "Prediction Reliability": f"{f_res.get('prediction_confidence', 0.0)}%"
                })
            
            st.plotly_chart(create_risk_trend_chart(hist_years, hist_scores, forecast_years, forecast_scores, pred_city), use_container_width=True)
            
            if len(multi_year_table) > 0:
                with st.expander("📅 View Year-by-Year Forecast Table", expanded=True):
                    st.dataframe(pd.DataFrame(multi_year_table).set_index("Year"), use_container_width=True)
            
            st.markdown("#### 🔍 Why is this area predicted to have this risk?")
            st.caption("Top factors influencing the ML prediction (based on Random Forest feature importances):")
            wcol1, wcol2 = st.columns([1,1])
            with wcol1:
                for factor in res["why_factors"]:
                    st.markdown(f'<div class="reason-item"><b>{factor["factor_name"]}</b> ({factor["impact_level"]}): {factor["importance_pct"]:.1f}% influence</div>', unsafe_allow_html=True)
            with wcol2:
                # Compare historical vs projected values
                st.write("**What Changed Recently (Projected vs Historical)?**")
                latest_row = city_hist.iloc[-1]
                changes = []
                for factor in res["why_factors"]:
                    f = factor["factor_name"].replace(" ", "_")
                    old_v = latest_row.get(f, 0.0)
                    new_v = factor["value"]
                    if type(old_v) in [int, float] and type(new_v) in [int, float] and old_v > 0:
                        chg = ((new_v - old_v)/old_v)*100
                        arrow_c = "↑" if chg > 0 else "↓"
                        changes.append(f"- **{factor['factor_name']}**: {old_v:.2f} → {new_v:.2f} ({arrow_c} {abs(chg):.1f}%)")
                    else:
                        changes.append(f"- **{factor['factor_name']}**: {old_v:.2f} → {new_v:.2f}")
                
                for c in changes:
                    st.markdown(c)
                    
            if diff > 10:
                st.markdown("#### 🚨 Early Warning Alert")
                st.error(f"**Rapid Risk Increase Detected:** The projected Danger Score for {pred_city} is increasing substantially (+{diff_pct:.1f}%).")
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
    st.subheader("📊 District Crime Distribution & Interactive Area Comparison")
    
    st.markdown("#### 🔄 Compare Areas (Risk Forecast & Crime Analytics)")
    compare_areas = st.multiselect("Select 2-5 Areas for Head-to-Head Comparison:", sorted(df_final["City"].unique()), default=[sorted(df_final["City"].unique())[0], sorted(df_final["City"].unique())[1]] if len(df_final["City"].unique()) > 1 else None)
    
    if len(compare_areas) >= 2:
        compare_data = []
        max_yr = df_final["Year"].max()
        pred_yr = max_yr + 1
        for c in compare_areas:
            hist_rows = df_final[(df_final["City"] == c) & (df_final["Year"] == max_yr)]
            hist_row = hist_rows.iloc[0] if len(hist_rows) > 0 else None
            if hist_row is not None:
                f_res = predict_danger_for_user_input(df_final, c, pred_yr)
                pred_s = f_res.get("predicted_danger_score", "N/A")
                rel_s = f_res.get("prediction_confidence", "N/A")
                compare_data.append({
                    "Area": c,
                    "Current Danger Score": round(float(hist_row.get("Danger_Score", 0.0)), 1),
                    f"Predicted Risk ({pred_yr})": pred_s,
                    "Risk Level": f_res.get("risk_level", "-"),
                    "Total Crime": int(hist_row.get("Total_Crime_Against_Women", 0)),
                    "Crime Trend": hist_row.get("Recent_Crime_Trend", "-"),
                    "Anomaly Status": hist_row.get("Anomaly_Status", "-"),
                    "Prediction Reliability (%)": rel_s
                })
        if compare_data:
            st.dataframe(pd.DataFrame(compare_data), use_container_width=True)
            
    st.markdown("---")
    
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
    
    st.info(f"🕰️ **Time-Aware Chronological Split:** Training uses historical data ({min(ml_metrics.get('train_years',[2014]))}–{max(ml_metrics.get('train_years',[2020]))}) with {ml_metrics.get('train_records',0)} records to predict Future Danger Score. Evaluation is performed strictly on unseen dataset from {ml_metrics.get('test_years',[2021])} with {ml_metrics.get('test_records',0)} records to prevent target leakage.")

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
