import sys
sys.path.insert(0, '.')

from src.data_loader import load_crime_data
from src.preprocessing import clean_and_preprocess_data, generate_data_quality_report
from src.feature_engineering import compute_derived_features
from src.anomaly_detection import run_isolation_forest
from src.danger_score import calculate_danger_scores, DEFAULT_WEIGHTS
from src.confidence_score import calculate_confidence_score
from src.clustering import run_dbscan_clustering
from src.model_training import train_and_evaluate_models
from src.prediction import predict_danger_for_user_input
import os

data_path = os.path.join('data', 'ncrb_district_crime_women.csv')
df_raw, _ = load_crime_data(data_path)
df_clean = clean_and_preprocess_data(df_raw)
dq = generate_data_quality_report(df_raw, df_clean)
df_features = compute_derived_features(df_clean)
df_anomaly, _ = run_isolation_forest(df_features, contamination=0.1)
df_scored, _ = calculate_danger_scores(df_anomaly, DEFAULT_WEIGHTS)
df_final = calculate_confidence_score(df_scored, dq)
df_final, _ = run_dbscan_clustering(df_final)
model_obj, scaler_obj, ml_metrics, _ = train_and_evaluate_models(df_final)

print('=== MODEL METRICS ===')
print('Best Model:', ml_metrics['best_model_name'])
print('Test MAE:', ml_metrics['test_mae'])
print('Test RMSE:', ml_metrics['test_rmse'])
print('Test R2:', ml_metrics['test_r2'])
print('Train years:', sorted(ml_metrics['train_years']))
print('Test years:', sorted(ml_metrics['test_years']))
print('Train records:', ml_metrics['train_records'])
print('Test records:', ml_metrics['test_records'])
print()

sample_city = df_final['City'].iloc[0]
max_year = int(df_final['Year'].max())
print(f'Sample city: {sample_city}  |  Latest year in data: {max_year}')
print()

for yr in [max_year + 1, 2027, 2030, 2035]:
    if yr <= max_year:
        yr = max_year + 1
    res = predict_danger_for_user_input(df_final, sample_city, yr)
    if res['status'] == 'success':
        print(f'{yr}: Score={res["predicted_danger_score"]}, Risk={res["risk_level"]}, Reliability={res["prediction_confidence"]}%')
    else:
        print(f'{yr}: ERROR -> {res["message"]}')
