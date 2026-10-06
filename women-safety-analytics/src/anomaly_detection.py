import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ANOMALY_NOTE = "Anomaly signal indicates that an area's crime pattern differs statistically from baseline observations; it does not directly equal guaranteed danger."

def run_isolation_forest(df, contamination=0.1, random_state=42):
    """
    Runs Isolation Forest algorithm to detect unusual crime patterns.
    Saves model artifact to models/isolation_forest.pkl.
    """
    df_anomaly = df.copy()
    
    feature_cols = [c for c in ["Normalized_Crime_Intensity", "Serious_Crime_Ratio", "Crime_Concentration", "Relative_Crime_Intensity", "YoY_Change_Pct"] if c in df_anomaly.columns]
    if not feature_cols:
        feature_cols = ["Total_Crime_Against_Women"]
        
    X = df_anomaly[feature_cols].fillna(0).values
    
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    
    preds = model.fit_predict(X)  # 1 for normal, -1 for anomaly
    raw_scores = model.decision_function(X) # lower score means more anomalous
    
    # Scale raw decision scores to 0-100 Anomaly Intensity (higher = more anomalous)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s > min_s:
        anomaly_intensity = 100 * (1 - (raw_scores - min_s) / (max_s - min_s))
    else:
        anomaly_intensity = np.zeros_like(raw_scores)
        
    df_anomaly["Anomaly_Flag"] = preds
    df_anomaly["Anomaly_Status"] = np.where(preds == -1, "Anomalous", "Normal")
    df_anomaly["Anomaly_Score"] = np.round(anomaly_intensity, 1)
    
    # Save trained model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "isolation_forest.pkl")
    joblib.dump(model, model_path)
    
    return df_anomaly, model

def get_anomaly_summary(df_anomaly):
    """
    Summarizes anomaly detection results.
    """
    total_obs = len(df_anomaly)
    anomalies_cnt = (df_anomaly["Anomaly_Status"] == "Anomalous").sum()
    pct = round((anomalies_cnt / total_obs) * 100, 1) if total_obs > 0 else 0.0
    
    anomalies_df = df_anomaly[df_anomaly["Anomaly_Status"] == "Anomalous"].copy()
    
    return {
        "total_observations": total_obs,
        "detected_anomalies": int(anomalies_cnt),
        "anomaly_percentage": pct,
        "anomalies_table": anomalies_df,
        "responsible_use_note": ANOMALY_NOTE
    }
