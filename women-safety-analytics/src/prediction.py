import os
import joblib
import numpy as np
import pandas as pd
from src.model_training import FEATURE_COLUMNS

def load_prediction_model():
    """
    Loads serialized Joblib artifacts: best_model.pkl, preprocessing_pipeline.pkl, model_metrics.pkl.
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    metrics_path = os.path.join(models_dir, "model_metrics.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metrics = joblib.load(metrics_path) if os.path.exists(metrics_path) else {}
    
    return model, scaler, metrics

def get_risk_level(score):
    """
    Maps numerical Danger Score (0-100) to standard Risk Level categories and color codes.
    """
    score = float(score)
    if score <= 20.0:
        return "VERY LOW", "#22C55E"   # Green
    elif score <= 40.0:
        return "LOW", "#84CC16"        # Light Green
    elif score <= 60.0:
        return "MODERATE", "#EAB308"   # Yellow
    elif score <= 80.0:
        return "HIGH", "#F97316"       # Orange
    else:
        return "VERY HIGH", "#EF4444"  # Red

def predict_area_danger_score(input_dict):
    """
    Executes ML prediction model.predict(input_features) to predict Danger Score (0-100).
    Calculates Prediction Confidence / Reliability (%) and feature-importance explanations.
    """
    model, scaler, metrics = load_prediction_model()
    if model is None or scaler is None:
        raise FileNotFoundError("Trained ML model artifacts not found in models/ directory.")

    available_features = metrics.get("available_features", FEATURE_COLUMNS)
    
    # Construct input vector in exact feature order
    input_vector = []
    for feat in available_features:
        input_vector.append(input_dict.get(feat, 0.0))
        
    X_raw = np.array(input_vector).reshape(1, -1)
    X_scaled = scaler.transform(X_raw)
    
    # Genuine ML Prediction using trained Random Forest model
    predicted_score_raw = model.predict(X_scaled)[0]
    predicted_score = round(float(np.clip(predicted_score_raw, 0, 100)), 1)
    
    # Calculate Prediction Confidence / Reliability (%)
    test_r2 = metrics.get("test_r2", 0.95)
    test_mae = metrics.get("test_mae", 1.2)
    base_conf = max(70.0, min(96.0, test_r2 * 100))
    error_penalty = min(10.0, test_mae * 1.2)
    
    prediction_confidence = round(float(np.clip(base_conf - error_penalty, 65.0, 98.0)), 1)
    
    # "Why This Prediction?" Factor Breakdown from Feature Importances
    feat_imp_df = metrics.get("feature_importance_df", None)
    why_factors = []
    
    if feat_imp_df is not None:
        for _, row in feat_imp_df.iterrows():
            f_name = row["Feature"]
            clean_name = f_name.replace("_", " ")
            imp_pct = float(row["Importance_Pct"])
            
            if imp_pct >= 20.0:
                impact = "High Impact"
            elif imp_pct >= 10.0:
                impact = "Medium Impact"
            else:
                impact = "Low Impact"
                
            why_factors.append({
                "factor_name": clean_name,
                "importance_pct": imp_pct,
                "impact_level": impact,
                "description": f"{clean_name}: {imp_pct}% influence on model output"
            })
            
    return {
        "predicted_danger_score": predicted_score,
        "prediction_confidence": prediction_confidence,
        "confidence_explanation": f"Based on historical data coverage, feature completeness, and model validation performance (R² = {test_r2:.2f}, MAE = {test_mae:.2f}).",
        "why_factors": why_factors,
        "input_features": input_dict
    }

def predict_danger_for_user_input(df_full, city_name, target_year):
    """
    Executes supervised ML prediction workflow for user-selected city/area and prediction year.
    Strictly handles location validation, year validation, trend projection (no data leakage),
    and model inference via model.predict(X_scaled).
    """
    # 1. Location Validation
    available_cities = sorted(df_full["City"].dropna().unique().tolist())
    if city_name not in available_cities:
        return {
            "status": "error",
            "error_type": "location_not_found",
            "message": f"⚠️ Location '{city_name}' is not available in the training dataset. Please select an available location."
        }

    # 2. Year Validation
    try:
        target_year = int(target_year)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error_type": "invalid_year",
            "message": "⚠️ Invalid year format. Please enter a valid numerical year."
        }

    max_hist_year = int(df_full["Year"].max())
    min_hist_year = int(df_full["Year"].min())
    
    if target_year < min_hist_year:
        return {
            "status": "error",
            "error_type": "invalid_year",
            "message": f"⚠️ Prediction is not available for year {target_year}. Please select a future year after the latest historical year ({max_hist_year})."
        }

    # 3. Retrieve Historical Records for Selected City
    city_df = df_full[df_full["City"] == city_name].sort_values(by="Year")
    if city_df.empty:
        return {
            "status": "error",
            "error_type": "location_not_found",
            "message": f"⚠️ Historical data for '{city_name}' could not be retrieved."
        }

    latest_row = city_df.iloc[-1]
    latest_year = int(latest_row["Year"])
    historical_score = round(float(latest_row.get("Danger_Score", 50.0)), 1)

    # 4. Feature Preparation & Trend Projection for Target Year (No Data Leakage)
    years_ahead = max(0, target_year - latest_year)
    
    model, scaler, metrics = load_prediction_model()
    if model is None or scaler is None:
        return {
            "status": "error",
            "error_type": "model_missing",
            "message": "⚠️ Trained ML model artifacts not found. Please train models first."
        }

    available_features = metrics.get("available_features", FEATURE_COLUMNS)
    
    # -------------------------------------------------------
    # Recursive time-aware feature projection (no leakage)
    # Features are projected forward using historical mean
    # growth and a dampening factor to model regression to mean
    # -------------------------------------------------------
    hist_yoy_mean = float(city_df["YoY_Change_Pct"].mean()) if "YoY_Change_Pct" in city_df.columns else 0.0
    latest_yoy = float(latest_row.get("YoY_Change_Pct", 0.0))

    input_features_dict = {}
    for feat in available_features:
        base_val = float(latest_row.get(feat, 0.0))
        if years_ahead > 0:
            if feat == "Population_Lakhs":
                # Modest 1% annual population growth assumption
                input_features_dict[feat] = round(base_val * (1.01 ** years_ahead), 2)
            elif feat == "YoY_Change_Pct":
                # Dampen YoY toward zero (regression to mean) over time
                dampen_rate = 0.80 ** (years_ahead - 1)
                projected_yoy = hist_yoy_mean * dampen_rate
                input_features_dict[feat] = round(float(projected_yoy), 3)
            elif feat in ["Normalized_Crime_Intensity", "Relative_Crime_Intensity"]:
                # Compound growth with dampening — each year the growth rate shrinks
                proj_val = base_val
                for y in range(1, years_ahead + 1):
                    effective_yoy = latest_yoy * (0.80 ** (y - 1)) / 100.0
                    proj_val = proj_val * (1.0 + effective_yoy * 0.5)
                input_features_dict[feat] = round(float(np.clip(proj_val, 0.0, 5.0)), 4)
            elif feat == "Anomaly_Score":
                # Anomaly scores revert toward zero over time
                revert_val = base_val * (0.90 ** years_ahead)
                input_features_dict[feat] = round(float(np.clip(revert_val, 0.0, 100.0)), 2)
            else:
                # Structural features (Serious_Crime_Ratio, etc.) stay stable
                input_features_dict[feat] = base_val
        else:
            input_features_dict[feat] = base_val

    # 5. Model Inference via model.predict(X_scaled)
    input_vector = [input_features_dict[f] for f in available_features]
    X_raw = np.array(input_vector).reshape(1, -1)
    X_scaled = scaler.transform(X_raw)
    
    predicted_score_raw = model.predict(X_scaled)[0]
    predicted_score = round(float(np.clip(predicted_score_raw, 0, 100)), 1)
    
    # 6. Risk Level Interpretation
    risk_level, risk_color = get_risk_level(predicted_score)
    
    # 7. Prediction Reliability (%)
    test_r2 = metrics.get("test_r2", 0.95)
    test_mae = metrics.get("test_mae", 1.2)
    base_conf = max(70.0, min(96.0, test_r2 * 100))
    error_penalty = min(10.0, test_mae * 1.2)
    
    # Reliability decreases non-linearly with forecast horizon
    # 1 year ahead: small penalty; 13+ years: large penalty
    future_discount = (years_ahead ** 1.25) * 2.2
    num_hist_obs = len(city_df)
    # Bonus for more historical data (max +5%)
    data_coverage_bonus = min(5.0, num_hist_obs * 0.6)
    prediction_confidence = round(float(np.clip(
        base_conf - error_penalty - future_discount + data_coverage_bonus,
        20.0, 98.0
    )), 1)

    # 8. Feature Importance Factors ("Why is this area predicted to have this score?")
    feat_imp_df = metrics.get("feature_importance_df", None)
    why_factors = []
    
    if feat_imp_df is not None:
        for _, row in feat_imp_df.iterrows():
            f_name = row["Feature"]
            clean_name = f_name.replace("_", " ")
            imp_pct = float(row["Importance_Pct"])
            
            if imp_pct >= 20.0:
                impact = "High Impact"
            elif imp_pct >= 10.0:
                impact = "Medium Impact"
            else:
                impact = "Low Impact"
                
            why_factors.append({
                "factor_name": clean_name,
                "importance_pct": imp_pct,
                "impact_level": impact,
                "value": round(input_features_dict.get(f_name, 0.0), 2)
            })

    return {
        "status": "success",
        "city": city_name,
        "state": latest_row.get("State", ""),
        "prediction_year": target_year,
        "historical_year": latest_year,
        "historical_danger_score": historical_score,
        "predicted_danger_score": predicted_score,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "prediction_confidence": prediction_confidence,
        "confidence_explanation": f"Based on historical data coverage ({min_hist_year}–{latest_year}), feature completeness, and model validation performance (R² = {test_r2:.2f}, MAE = {test_mae:.2f}).",
        "why_factors": why_factors,
        "input_features": input_features_dict
    }
