import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Column Mapping Layer: Matches messy or variant column names from NCRB reports to canonical names
COLUMN_MAPPING_RULES = {
    "state": "State",
    "state/ut": "State",
    "district": "City",
    "city": "City",
    "metropolitan_city": "City",
    "area": "City",
    "year": "Year",
    "rape": "Rape",
    "rape (sec 376 ipc)": "Rape",
    "kidnapping_and_abduction": "Kidnapping_and_Abduction",
    "kidnapping & abduction": "Kidnapping_and_Abduction",
    "dowry_deaths": "Dowry_Deaths",
    "dowry deaths (sec 304b ipc)": "Dowry_Deaths",
    "assault_on_women": "Assault_on_Women",
    "assault on women with intent to outrage her modesty": "Assault_on_Women",
    "cruelty_by_husband_or_relatives": "Cruelty_by_Husband_or_Relatives",
    "cruelty by husband or his relatives": "Cruelty_by_Husband_or_Relatives",
    "sexual_harassment": "Sexual_Harassment",
    "insult to modesty of women": "Sexual_Harassment",
    "cyber_crime": "Cyber_Crime",
    "cyber crime against women": "Cyber_Crime",
    "other_crimes_against_women": "Other_Crimes_Against_Women",
    "total_crime_against_women": "Total_Crime_Against_Women",
    "total crimes against women": "Total_Crime_Against_Women",
    "total_women_crimes": "Total_Crime_Against_Women",
    "population_lakhs": "Population_Lakhs",
    "population (in lakhs)": "Population_Lakhs",
    "latitude": "Latitude",
    "lat": "Latitude",
    "longitude": "Longitude",
    "lon": "Longitude",
    "lng": "Longitude"
}

def standardize_column_names(df):
    """
    Applies column mapping layer to match complex/varied NCRB column names.
    Deduplicates target column names.
    """
    mapped_cols = {}
    
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean == "district" and "city" in [c.lower() for c in df.columns]:
            mapped_cols[col] = "District_Name"
        elif col_clean in COLUMN_MAPPING_RULES:
            mapped_cols[col] = COLUMN_MAPPING_RULES[col_clean]
        else:
            matched = False
            for rule_key, canonical_name in COLUMN_MAPPING_RULES.items():
                if rule_key in col_clean:
                    mapped_cols[col] = canonical_name
                    matched = True
                    break
            if not matched:
                mapped_cols[col] = col

    df_renamed = df.rename(columns=mapped_cols)
    # Remove duplicate columns if any remain
    df_renamed = df_renamed.loc[:, ~df_renamed.columns.duplicated()]
    return df_renamed, mapped_cols

def clean_and_preprocess_data(df):
    """
    Performs data cleaning: missing value handling, duplicate removal, numeric conversion.
    """
    df_clean, mapping_used = standardize_column_names(df.copy())
    
    # Drop exact duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Expected numeric crime categories
    numeric_cols = [
        "Rape", "Kidnapping_and_Abduction", "Dowry_Deaths", "Assault_on_Women",
        "Cruelty_by_Husband_or_Relatives", "Sexual_Harassment", "Cyber_Crime",
        "Other_Crimes_Against_Women", "Total_Crime_Against_Women", "Population_Lakhs",
        "Latitude", "Longitude"
    ]
    
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
    # Ensure Total_Crime_Against_Women is calculated if missing
    if "Total_Crime_Against_Women" not in df_clean.columns or (df_clean["Total_Crime_Against_Women"] == 0).all():
        crime_sub_cols = [c for c in ["Rape", "Kidnapping_and_Abduction", "Dowry_Deaths", "Assault_on_Women", 
                                      "Cruelty_by_Husband_or_Relatives", "Sexual_Harassment", "Cyber_Crime", "Other_Crimes_Against_Women"] if c in df_clean.columns]
        if crime_sub_cols:
            df_clean["Total_Crime_Against_Women"] = df_clean[crime_sub_cols].sum(axis=1)
            
    return df_clean

def scale_features(df, feature_cols):
    """
    Standardizes feature vectors for ML models using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[feature_cols])
    return scaled_matrix, scaler

def generate_data_quality_report(df_raw, df_clean):
    """
    Generates a structured data quality audit dict.
    """
    total_cells = df_raw.shape[0] * df_raw.shape[1]
    missing_cells = df_raw.isnull().sum().sum()
    completeness_pct = round(((total_cells - missing_cells) / total_cells) * 100, 2) if total_cells > 0 else 100.0
    
    years = sorted(df_clean["Year"].dropna().unique().astype(int).tolist()) if "Year" in df_clean.columns else []
    cities = df_clean["City"].dropna().unique().tolist() if "City" in df_clean.columns else []
    states = df_clean["State"].dropna().unique().tolist() if "State" in df_clean.columns else []
    
    return {
        "total_rows": df_clean.shape[0],
        "total_columns": df_clean.shape[1],
        "raw_missing_values": int(missing_cells),
        "duplicate_rows_removed": int(df_raw.shape[0] - df_clean.shape[0]),
        "data_completeness_pct": completeness_pct,
        "available_years": years,
        "total_cities": len(cities),
        "cities_list": cities,
        "total_states": len(states)
    }
