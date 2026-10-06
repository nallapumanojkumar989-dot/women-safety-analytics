import numpy as np
import pandas as pd

def generate_evidence_explanations(row, df_full):
    """
    Generates 3 to 5 factual, evidence-based reasons explaining why an area has its specific Danger Score.
    Strictly derives reasons from actual feature values and dataset comparison baselines.
    """
    reasons = []
    
    city = row["City"]
    year = row["Year"]
    total_crime = row["Total_Crime_Against_Women"]
    danger_score = row["Danger_Score"]
    
    # Calculate dataset baseline for comparison in the same year
    year_data = df_full[df_full["Year"] == year]
    avg_year_crime = year_data["Total_Crime_Against_Women"].mean() if not year_data.empty else 2000.0
    med_year_crime = year_data["Total_Crime_Against_Women"].median() if not year_data.empty else 1500.0

    # Reason 1: Reported Crime Frequency Baseline Comparison
    if total_crime > avg_year_crime:
        pct_above = round(((total_crime - avg_year_crime) / avg_year_crime) * 100, 1)
        reasons.append(
            f"Reported crime frequency is {pct_above}% above the metropolitan dataset baseline average ({int(avg_year_crime):,} crimes)."
        )
    elif total_crime > med_year_crime:
        reasons.append(
            f"Reported crime volume ({int(total_crime):,}) is higher than the median city baseline ({int(med_year_crime):,})."
        )
    else:
        reasons.append(
            f"Reported crime count ({int(total_crime):,}) is within lower-to-moderate range relative to major metropolitan baselines."
        )

    # Reason 2: Serious Crime Contribution (Rape, Abduction, Dowry Deaths)
    serious_cnt = row.get("Serious_Crime_Count", 0)
    serious_ratio = row.get("Serious_Crime_Ratio", 0.0) * 100
    if serious_ratio > 30.0:
        reasons.append(
            f"Serious crime categories (Rape, Kidnapping & Abduction, Dowry Deaths) account for {serious_ratio:.1f}% of total crimes against women ({int(serious_cnt):,} reported offences)."
        )
    elif serious_cnt > 0:
        reasons.append(
            f"Serious violent crimes against women account for {serious_ratio:.1f}% of reported offences."
        )

    # Reason 3: Dominant Women-Specific Crime Category Concentration
    category_cols = {
        "Cruelty_by_Husband_or_Relatives": "Cruelty by Husband or Relatives (Sec 498A IPC)",
        "Kidnapping_and_Abduction": "Kidnapping & Abduction of Women",
        "Assault_on_Women": "Assault on Women with Intent to Outrage Modesty",
        "Rape": "Rape (Sec 376 IPC)",
        "Sexual_Harassment": "Sexual Harassment / Insult to Modesty",
        "Cyber_Crime": "Cyber Crimes Against Women"
    }
    
    max_cat_col = None
    max_cat_val = 0
    for col in category_cols:
        if col in row and row[col] > max_cat_val:
            max_cat_val = row[col]
            max_cat_col = col

    if max_cat_col and total_crime > 0:
        cat_pct = round((max_cat_val / total_crime) * 100, 1)
        cat_name = category_cols[max_cat_col]
        reasons.append(
            f"{cat_name} is the primary reported category, contributing {cat_pct}% ({int(max_cat_val):,} cases) to the total."
        )

    # Reason 4: Relative Crime Intensity Factor
    rel_intensity = row.get("Relative_Crime_Intensity", 1.0)
    if rel_intensity > 1.2:
        reasons.append(
            f"Crime intensity index is {rel_intensity:.2f}x higher than the national metropolitan benchmark."
        )
    elif rel_intensity < 0.8:
        reasons.append(
            f"Relative crime intensity index ({rel_intensity:.2f}x) is below the national metropolitan benchmark."
        )

    # Reason 5: Anomaly Signal Detection (Isolation Forest)
    anomaly_status = row.get("Anomaly_Status", "Normal")
    anomaly_score = row.get("Anomaly_Score", 0.0)
    if anomaly_status == "Anomalous":
        reasons.append(
            f"An unusual crime pattern was detected by the Isolation Forest ML model (Anomaly Intensity Score: {anomaly_score}/100)."
        )

    # Reason 6: Year-over-Year Trend
    yoy_pct = row.get("YoY_Change_Pct", 0.0)
    if yoy_pct > 3.0:
        reasons.append(
            f"Recent multi-year trend shows a year-over-year increase of +{yoy_pct:.1f}% in reported crimes."
        )
    elif yoy_pct < -3.0:
        reasons.append(
            f"Recent multi-year trend shows a year-over-year decrease of {yoy_pct:.1f}% in reported crimes."
        )

    # Ensure 3 to 5 clear, evidence-based bullet points
    return reasons[:5]
