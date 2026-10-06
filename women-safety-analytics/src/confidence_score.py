import numpy as np
import pandas as pd

CONFIDENCE_EXPLANATION = "Confidence represents the reliability of the available historical evidence and data completeness, NOT the probability of a future crime."

def calculate_confidence_score(df, data_quality_info):
    """
    Computes a transparent Risk Assessment Confidence Score (0-100%) for each observation in the dataset.
    """
    df_conf = df.copy()
    
    completeness_base = data_quality_info.get("data_completeness_pct", 95.0)
    available_years_cnt = len(data_quality_info.get("available_years", [2023]))
    
    # 1. Historical Coverage Factor (1 year = 60%, 2 years = 80%, 3+ years = 100%)
    year_factor = min(available_years_cnt / 3.0, 1.0) * 100
    
    # 2. Data Volume Factor (Higher reported volume gives higher sample stability)
    max_c = df_conf["Total_Crime_Against_Women"].max()
    volume_score = np.where(
        max_c > 0,
        np.clip(50 + (df_conf["Total_Crime_Against_Women"] / max_c) * 50, 50, 100),
        70.0
    )

    # 3. Model Stability Factor (Normal points = 95% stability, Anomalous = 80% stability)
    model_stability = np.where(df_conf.get("Anomaly_Status", "Normal") == "Normal", 95.0, 82.0)

    # Weighted Confidence Score
    conf_raw = (
        completeness_base * 0.35 +
        year_factor * 0.25 +
        volume_score * 0.20 +
        model_stability * 0.20
    )

    df_conf["Confidence_Score"] = np.round(np.clip(conf_raw, 40.0, 99.0), 1)
    
    return df_conf

def get_confidence_tooltip():
    return CONFIDENCE_EXPLANATION
