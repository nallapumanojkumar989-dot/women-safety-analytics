import pandas as pd
import numpy as np

def compute_derived_features(df):
    """
    Computes meaningful derived features for crime safety analytics.
    Clearly marks all generated features with derived status metadata.
    """
    df_feat = df.copy()
    
    # 1. Serious Crime Count [Derived Feature]
    # Serious crimes against women: Rape, Kidnapping & Abduction, Dowry Deaths
    serious_cols = [c for c in ["Rape", "Kidnapping_and_Abduction", "Dowry_Deaths"] if c in df_feat.columns]
    if serious_cols:
        df_feat["Serious_Crime_Count"] = df_feat[serious_cols].sum(axis=1)
    else:
        df_feat["Serious_Crime_Count"] = 0
        
    # 2. Serious Crime Ratio [Derived Feature]
    df_feat["Serious_Crime_Ratio"] = np.where(
        df_feat["Total_Crime_Against_Women"] > 0,
        df_feat["Serious_Crime_Count"] / df_feat["Total_Crime_Against_Women"],
        0.0
    )

    # 3. Crime Rate per Lakh [Derived Feature] (Calculated ONLY when population data exists)
    if "Population_Lakhs" in df_feat.columns and (df_feat["Population_Lakhs"] > 0).any():
        df_feat["Crime_Rate_Per_Lakh"] = np.where(
            df_feat["Population_Lakhs"] > 0,
            df_feat["Total_Crime_Against_Women"] / df_feat["Population_Lakhs"],
            np.nan
        )
    else:
        df_feat["Crime_Rate_Per_Lakh"] = np.nan

    # 4. Normalized Crime Intensity [Derived Feature] (0-1 min-max scaling per year)
    df_feat["Normalized_Crime_Intensity"] = 0.0
    for year in df_feat["Year"].unique():
        year_mask = (df_feat["Year"] == year)
        min_c = df_feat.loc[year_mask, "Total_Crime_Against_Women"].min()
        max_c = df_feat.loc[year_mask, "Total_Crime_Against_Women"].max()
        if max_c > min_c:
            df_feat.loc[year_mask, "Normalized_Crime_Intensity"] = (
                (df_feat.loc[year_mask, "Total_Crime_Against_Women"] - min_c) / (max_c - min_c)
            )
        else:
            df_feat.loc[year_mask, "Normalized_Crime_Intensity"] = 0.5

    # 5. Crime Category Diversity (Shannon Entropy) [Derived Feature]
    category_cols = [c for c in ["Rape", "Kidnapping_and_Abduction", "Dowry_Deaths", "Assault_on_Women", 
                                 "Cruelty_by_Husband_or_Relatives", "Sexual_Harassment", "Cyber_Crime", "Other_Crimes_Against_Women"] if c in df_feat.columns]
    
    def calculate_entropy(row):
        total = row["Total_Crime_Against_Women"]
        if total <= 0:
            return 0.0
        props = [row[c] / total for c in category_cols if row[c] > 0]
        if not props:
            return 0.0
        return -sum(p * np.log2(p) for p in props)

    df_feat["Crime_Category_Diversity"] = df_feat.apply(calculate_entropy, axis=1)

    # 6. Crime Concentration Ratio (Top 2 crime categories proportion) [Derived Feature]
    def calculate_concentration(row):
        total = row["Total_Crime_Against_Women"]
        if total <= 0:
            return 0.0
        vals = sorted([row[c] for c in category_cols], reverse=True)
        top2 = sum(vals[:2])
        return top2 / total

    df_feat["Crime_Concentration"] = df_feat.apply(calculate_concentration, axis=1)

    # 7. Relative Crime Intensity (Ratio to annual mean across all cities) [Derived Feature]
    df_feat["Relative_Crime_Intensity"] = 1.0
    for year in df_feat["Year"].unique():
        year_mask = (df_feat["Year"] == year)
        mean_c = df_feat.loc[year_mask, "Total_Crime_Against_Women"].mean()
        if mean_c > 0:
            df_feat.loc[year_mask, "Relative_Crime_Intensity"] = (
                df_feat.loc[year_mask, "Total_Crime_Against_Women"] / mean_c
            )

    # 8. YoY Change & Recent Crime Trend [Derived Feature]
    df_feat = df_feat.sort_values(by=["City", "Year"])
    df_feat["YoY_Change_Pct"] = 0.0
    
    for city in df_feat["City"].unique():
        city_mask = (df_feat["City"] == city)
        city_sub = df_feat.loc[city_mask]
        if len(city_sub) > 1:
            pct_change = city_sub["Total_Crime_Against_Women"].pct_change() * 100
            df_feat.loc[city_mask, "YoY_Change_Pct"] = pct_change.fillna(0.0)

    # Trend classification string
    def classify_trend(val):
        if val > 5.0:
            return "Increasing (+{:.1f}%)".format(val)
        elif val < -5.0:
            return "Decreasing ({:.1f}%)".format(val)
        else:
            return "Stable ({:+.1f}%)".format(val)

    df_feat["Recent_Crime_Trend"] = df_feat["YoY_Change_Pct"].apply(classify_trend)

    return df_feat

def get_derived_feature_labels():
    """
    Returns explicit labels marking engineered features vs raw NCRB fields.
    """
    return {
        "Serious_Crime_Count": "[Derived Feature] Total counts of Rape, Kidnapping, and Dowry Deaths",
        "Serious_Crime_Ratio": "[Derived Feature] Proportion of total crime categorized as severe offences",
        "Crime_Rate_Per_Lakh": "[Derived Feature] Crimes per 1,000,000 population (if population exists)",
        "Normalized_Crime_Intensity": "[Derived Feature] 0 to 1 min-max crime intensity relative to yearly baseline",
        "Crime_Category_Diversity": "[Derived Feature] Shannon Entropy measure of crime distribution across types",
        "Crime_Concentration": "[Derived Feature] Combined proportion of the top 2 highest reported crime types",
        "Relative_Crime_Intensity": "[Derived Feature] Ratio of city crime total compared to annual metropolitan average",
        "YoY_Change_Pct": "[Derived Feature] Year-over-Year percentage change in reported crimes against women",
        "Recent_Crime_Trend": "[Derived Feature] Categorical trend direction (Increasing, Decreasing, Stable)"
    }
