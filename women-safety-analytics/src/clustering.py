import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def run_dbscan_clustering(df, eps=0.5, min_samples=2, use_geographic=False):
    """
    Executes DBSCAN clustering on geographic coordinates or feature vectors.
    Saves trained model to models/dbscan_model.pkl.
    """
    df_cluster = df.copy()
    
    if use_geographic and "Latitude" in df_cluster.columns and "Longitude" in df_cluster.columns:
        # Geographic Clustering using Lat/Lon (convert degrees to radians for haversine metric or scaled Euclidean)
        coords = df_cluster[["Latitude", "Longitude"]].values
        scaler = StandardScaler()
        scaled_coords = scaler.fit_transform(coords)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(scaled_coords)
    else:
        # Feature-space Clustering using normalized crime vectors
        feature_cols = [c for c in ["Normalized_Crime_Intensity", "Serious_Crime_Ratio", "Crime_Concentration", "Relative_Crime_Intensity"] if c in df_cluster.columns]
        if not feature_cols:
            feature_cols = ["Total_Crime_Against_Women"]
            
        feat_matrix = df_cluster[feature_cols].fillna(0).values
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(feat_matrix)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(scaled_matrix)

    df_cluster["Cluster"] = cluster_labels
    df_cluster["Cluster_Label"] = df_cluster["Cluster"].apply(lambda x: f"Cluster {x+1}" if x != -1 else "Noise (Outlier)")
    
    # Save model artifact
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "dbscan_model.pkl")
    joblib.dump(dbscan, model_path)
    
    return df_cluster, dbscan

def summarize_clusters(df_clustered):
    """
    Generates summary statistics for each detected DBSCAN cluster.
    """
    summary = []
    category_cols = [c for c in ["Rape", "Kidnapping_and_Abduction", "Dowry_Deaths", "Assault_on_Women", 
                                 "Cruelty_by_Husband_or_Relatives", "Sexual_Harassment", "Cyber_Crime"] if c in df_clustered.columns]

    for cluster_id, group in df_clustered.groupby("Cluster"):
        cluster_name = f"Cluster {cluster_id + 1}" if cluster_id != -1 else "Noise (Unclustered)"
        num_areas = len(group)
        avg_crime = group["Total_Crime_Against_Women"].mean()
        avg_danger = group["Danger_Score"].mean() if "Danger_Score" in group.columns else 0.0
        
        # Determine dominant crime category
        if category_cols:
            cat_sums = group[category_cols].sum()
            dominant_cat = cat_sums.idxmax() if not cat_sums.empty else "N/A"
            dominant_cat_clean = dominant_cat.replace("_", " ")
        else:
            dominant_cat_clean = "N/A"
            
        summary.append({
            "Cluster_ID": cluster_id,
            "Cluster_Name": cluster_name,
            "Area_Count": num_areas,
            "Avg_Total_Crime": round(avg_crime, 1),
            "Avg_Danger_Score": round(avg_danger, 1),
            "Dominant_Crime_Category": dominant_cat_clean,
            "Cities": ", ".join(group["City"].unique()[:5])
        })
        
    return pd.DataFrame(summary)
