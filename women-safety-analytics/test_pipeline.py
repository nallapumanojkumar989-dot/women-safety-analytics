import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_crime_data, get_data_metadata
from src.preprocessing import clean_and_preprocess_data, generate_data_quality_report
from src.feature_engineering import compute_derived_features
from src.clustering import run_dbscan_clustering, summarize_clusters
from src.anomaly_detection import run_isolation_forest, get_anomaly_summary
from src.danger_score import calculate_danger_scores
from src.confidence_score import calculate_confidence_score
from src.explanations import generate_evidence_explanations
from src.model_training import train_and_evaluate_models, FEATURE_COLUMNS
from src.prediction import predict_area_danger_score, load_prediction_model

def test_full_pipeline():
    print("=== STARTING ML PREDICTION PIPELINE VERIFICATION TEST ===")
    
    # 1. Dataset Loading (Large NCRB Dataset)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ncrb_district_crime_women.csv")
    df_raw, attribution = load_crime_data(csv_path)
    meta = get_data_metadata(csv_path)
    print(f"[OK] Step 1: Large Dataset Loaded. Rows: {len(df_raw)}, Cols: {len(df_raw.columns)}, Type: {meta['type']}")
    assert len(df_raw) >= 500, "Dataset should have over 500 records"
    
    # 2. Preprocessing & Quality Audit
    df_clean = clean_and_preprocess_data(df_raw)
    dq_report = generate_data_quality_report(df_raw, df_clean)
    print(f"[OK] Step 2: Preprocessing Complete. Clean Rows: {len(df_clean)}, Data Completeness: {dq_report['data_completeness_pct']}%")
    
    # 3. Derived Feature Engineering
    df_feat = compute_derived_features(df_clean)
    print(f"[OK] Step 3: Derived Features Computed. Total Columns: {len(df_feat.columns)}")
    
    # 4. Isolation Forest Anomaly Detection
    df_anom, iso_model = run_isolation_forest(df_feat)
    anom_summary = get_anomaly_summary(df_anom)
    print(f"[OK] Step 4: Isolation Forest Complete. Detected Anomalies: {anom_summary['detected_anomalies']}")
    
    # 5. Target Danger Score Calculation (Historical Target)
    df_scored, weights = calculate_danger_scores(df_anom)
    print(f"[OK] Step 5: Target Danger Scores Calculated. Range: {df_scored['Danger_Score'].min()} to {df_scored['Danger_Score'].max()}")
    
    # 6. Confidence Score Calculation
    df_final = calculate_confidence_score(df_scored, dq_report)
    print(f"[OK] Step 6: Confidence Scores Calculated. Avg Confidence: {round(df_final['Confidence_Score'].mean(), 1)}%")
    
    # 7. Supervised ML Training, 5-Fold Cross Validation & Model Comparison
    best_model, scaler, ml_metrics, actual_vs_pred_df = train_and_evaluate_models(df_final)
    print(f"[OK] Step 7: Supervised ML Models Trained.")
    print(f"     Selected Model: {ml_metrics['best_model_name']}")
    print(f"     5-Fold CV Val MAE: {ml_metrics['cv_val_mae']}, Test MAE: {ml_metrics['test_mae']}, Test R2: {ml_metrics['test_r2']}")
    print(f"     Possible Overfitting: {ml_metrics['possible_overfitting']}")
    
    # 8. Model Prediction Execution (model.predict)
    sample_input = {f: df_final.iloc[0].get(f, 0.0) for f in FEATURE_COLUMNS}
    pred_res = predict_area_danger_score(sample_input)
    print(f"[OK] Step 8: ML model.predict() Executed.")
    print(f"     Predicted Danger Score: {pred_res['predicted_danger_score']} / 100")
    print(f"     Prediction Reliability: {pred_res['prediction_confidence']}%")
    print(f"     Why Factors Extracted: {len(pred_res['why_factors'])} factors")
    
    # 9. DBSCAN Clustering Verification
    df_clustered, dbscan_model = run_dbscan_clustering(df_final, eps=0.6, min_samples=2)
    cluster_sum = summarize_clusters(df_clustered)
    print(f"[OK] Step 9: DBSCAN Clustering Verified. Clusters Formed: {len(cluster_sum)}")
    
    # 10. Persistence Artifact Verification
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    assert os.path.exists(os.path.join(models_dir, "best_model.pkl")), "best_model.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "preprocessing_pipeline.pkl")), "preprocessing_pipeline.pkl missing"
    assert os.path.exists(os.path.join(models_dir, "model_metrics.pkl")), "model_metrics.pkl missing"
    print(f"[OK] Step 10: Saved Joblib Model Artifacts Verified.")
    
    print("\n[SUCCESS] ALL ML PREDICTION PIPELINE TESTS PASSED!")

if __name__ == "__main__":
    test_full_pipeline()
