import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "crime_frequency": 0.25,      # 25%
    "women_crime_ratio": 0.20,    # 20%
    "serious_crime_ratio": 0.20,  # 20%
    "relative_intensity": 0.15,   # 15%
    "recent_trend": 0.10,         # 10%
    "anomaly_signal": 0.10        # 10%
}

def calculate_danger_scores(df, custom_weights=None):
    """
    Computes a transparent, explainable Danger Score (0-100) for every area observation.
    Returns dataframe with Danger_Score and detailed component breakdowns.
    """
    df_score = df.copy()
    weights = custom_weights if custom_weights is not None else DEFAULT_WEIGHTS
    
    # Normalize weights so they sum to 1.0
    w_total = sum(weights.values())
    w = {k: v / w_total for k, v in weights.items()}
    
    # Component 1: Crime Frequency / Normalized Intensity (0-100)
    norm_freq = df_score["Normalized_Crime_Intensity"] * 100
    
    # Component 2: Women-related Crime Ratio (100% since dataset is women specific, or normalized intensity)
    # We measure normalized crime count relative to population or maximum observed
    max_women_crime = df_score["Total_Crime_Against_Women"].max()
    women_ratio_score = np.where(
        max_women_crime > 0,
        (df_score["Total_Crime_Against_Women"] / max_women_crime) * 100,
        50.0
    )

    # Component 3: Serious Crime Ratio (Rape, Kidnapping, Dowry Deaths proportion) (0-100)
    serious_ratio_score = df_score["Serious_Crime_Ratio"] * 100

    # Component 4: Relative Crime Intensity (compared to annual baseline) (0-100)
    # Relative intensity of 1.0 = mean (50 score). Range 0 to 3 -> 0 to 100 score.
    rel_intensity_score = np.clip((df_score["Relative_Crime_Intensity"] / 2.5) * 100, 0, 100)

    # Component 5: Recent YoY Trend (0-100)
    # Positive growth = higher danger score. -50% to +50% mapped to 0-100.
    yoy_score = np.clip(50 + df_score["YoY_Change_Pct"] * 1.5, 0, 100)

    # Component 6: Anomaly Signal (0-100)
    anomaly_score = df_score["Anomaly_Score"] if "Anomaly_Score" in df_score.columns else np.zeros(len(df_score))

    # Calculate weighted Danger Score
    danger_raw = (
        norm_freq * w["crime_frequency"] +
        women_ratio_score * w["women_crime_ratio"] +
        serious_ratio_score * w["serious_crime_ratio"] +
        rel_intensity_score * w["relative_intensity"] +
        yoy_score * w["recent_trend"] +
        anomaly_score * w["anomaly_signal"]
    )
    
    df_score["Danger_Score"] = np.round(np.clip(danger_raw, 0, 100), 1)

    # Store component score contributions for feature contribution bar charts
    df_score["Contrib_Crime_Frequency"] = np.round(norm_freq * w["crime_frequency"], 1)
    df_score["Contrib_Women_Crime"] = np.round(women_ratio_score * w["women_crime_ratio"], 1)
    df_score["Contrib_Serious_Crime"] = np.round(serious_ratio_score * w["serious_crime_ratio"], 1)
    df_score["Contrib_Relative_Intensity"] = np.round(rel_intensity_score * w["relative_intensity"], 1)
    df_score["Contrib_Recent_Trend"] = np.round(yoy_score * w["recent_trend"], 1)
    df_score["Contrib_Anomaly_Signal"] = np.round(anomaly_score * w["anomaly_signal"], 1)

    return df_score, w

def get_area_danger_breakdown(row, weights):
    """
    Returns breakdown dictionary for a specific area row.
    """
    total = row["Danger_Score"] if "Danger_Score" in row else 1.0
    total = max(total, 1.0)
    
    contribs = {
        "Crime Frequency": row.get("Contrib_Crime_Frequency", 0.0),
        "Women-related Crime": row.get("Contrib_Women_Crime", 0.0),
        "Serious Crime": row.get("Contrib_Serious_Crime", 0.0),
        "Crime Intensity": row.get("Contrib_Relative_Intensity", 0.0),
        "Recent Trend": row.get("Contrib_Recent_Trend", 0.0),
        "Anomaly Signal": row.get("Contrib_Anomaly_Signal", 0.0)
    }
    
    # Calculate percentage contribution to the Danger Score
    pct_contribs = {k: round((v / total) * 100, 1) for k, v in contribs.items()}
    top_factor = max(pct_contribs, key=pct_contribs.get)
    
    return {
        "danger_score": row["Danger_Score"],
        "contributions": contribs,
        "percentage_contributions": pct_contribs,
        "primary_concern": f"High {top_factor.lower()} impact",
        "top_factor": top_factor
    }
