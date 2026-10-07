import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_COLUMNS = [
    "Normalized_Crime_Intensity",
    "Serious_Crime_Ratio",
    "Crime_Category_Diversity",
    "Crime_Concentration",
    "Relative_Crime_Intensity",
    "YoY_Change_Pct",
    "Anomaly_Score",
    "Population_Lakhs"
]

def prepare_time_series_data(df):
    """
    Shifts the 'Danger_Score' target variable by -1 year for each City to predict NEXT year's score.
    Prevents target leakage by predicting Future Danger Score based on Current Features.
    """
    df_ts = df.copy().sort_values(by=["City", "Year"])
    df_ts["Future_Danger_Score"] = df_ts.groupby("City")["Danger_Score"].shift(-1)
    return df_ts

def train_and_evaluate_models(df):
    """
    Trains, compares, and evaluates supervised ML regression models to predict Future Danger Score.
    Uses Chronological Split.
    """
    df_ts = prepare_time_series_data(df)
    
    available_features = [c for c in FEATURE_COLUMNS if c in df_ts.columns]
    
    # We will train and test only where Future_Danger_Score is not null
    df_ml = df_ts.dropna(subset=["Future_Danger_Score"]).copy()
    
    # Time-aware split: latest available training year becomes TEST set, older years become TRAIN set
    available_years = sorted(df_ml["Year"].unique())
    if len(available_years) < 2:
        # Fallback to random split if not enough years
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_ml, test_size=0.2, random_state=42)
    else:
        test_year = available_years[-1]
        train_df = df_ml[df_ml["Year"] < test_year]
        test_df = df_ml[df_ml["Year"] == test_year]
        
        # Ensure sufficient test size, else shift test_year back
        if len(test_df) < (0.05 * len(df_ml)) and len(available_years) > 2:
           test_year = available_years[-2]
           train_df = df_ml[df_ml["Year"] < test_year]
           test_df = df_ml[df_ml["Year"] >= test_year]

    train_years = train_df['Year'].unique().tolist()
    test_years = test_df['Year'].unique().tolist()

    X_train = train_df[available_features].fillna(0).values
    y_train = train_df["Future_Danger_Score"].values
    
    X_test = test_df[available_features].fillna(0).values
    y_test = test_df["Future_Danger_Score"].values
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define Candidate Models
    candidate_models = {
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    comparison_results = []
    trained_models = {}
    
    best_model_name = None
    best_val_mae = float('inf')
    best_model_obj = None
    
    for name, model in candidate_models.items():
        cv_scores = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'])
        val_mae = -np.mean(cv_scores['test_neg_mean_absolute_error'])
        val_rmse = -np.mean(cv_scores['test_neg_root_mean_squared_error'])
        val_r2 = np.mean(cv_scores['test_r2'])
        
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        y_train_pred = model.predict(X_train_scaled)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        y_test_pred = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        comparison_results.append({
            "Model": name,
            "CV_Val_MAE": round(val_mae, 2),
            "CV_Val_RMSE": round(val_rmse, 2),
            "CV_Val_R2": round(val_r2, 4),
            "Train_MAE": round(train_mae, 2),
            "Test_MAE": round(test_mae, 2),
            "Test_RMSE": round(test_rmse, 2),
            "Test_R2": round(test_r2, 4)
        })
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_name = name
            best_model_obj = model

    comparison_df = pd.DataFrame(comparison_results)
    best_model_info = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]
    
    rf_model = trained_models["Random Forest Regression"]
    rf_importances = rf_model.feature_importances_
    
    feat_imp_df = pd.DataFrame({
        "Feature": available_features,
        "Importance": rf_importances,
        "Importance_Pct": np.round(rf_importances * 100, 1)
    }).sort_values(by="Importance", ascending=False)
    
    y_best_pred = best_model_obj.predict(X_test_scaled)
    actual_vs_pred_df = pd.DataFrame({
        "Actual_Danger_Score": y_test,
        "Predicted_Danger_Score": np.round(y_best_pred, 1),
        "Absolute_Error": np.round(np.abs(y_test - y_best_pred), 1)
    })
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "preprocessing_pipeline.pkl"))
    
    metrics_summary = {
        "best_model_name": best_model_name,
        "test_mae": best_model_info["Test_MAE"],
        "test_rmse": best_model_info["Test_RMSE"],
        "test_r2": best_model_info["Test_R2"],
        "cv_val_mae": best_model_info["CV_Val_MAE"],
        "train_mae": best_model_info["Train_MAE"],
        "available_features": available_features,
        "comparison_df": comparison_df,
        "feature_importance_df": feat_imp_df,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "train_years": train_years,
        "test_years": test_years
    }
    
    joblib.dump(metrics_summary, os.path.join(models_dir, "model_metrics.pkl"))
    
    return best_model_obj, scaler, metrics_summary, actual_vs_pred_df
