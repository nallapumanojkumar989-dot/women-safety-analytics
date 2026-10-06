import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_THEME_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#E2E8F0", "family": "Inter, sans-serif"},
    "xaxis": {"gridcolor": "#334155", "zerolinecolor": "#334155"},
    "yaxis": {"gridcolor": "#334155", "zerolinecolor": "#334155"}
}

def create_danger_score_gauge(score, title="Danger Score", color_override=None):
    """
    Creates a Plotly gauge chart for the Danger Score (0-100).
    """
    gauge_color = color_override if color_override else (
        "#EF4444" if score >= 75 else ("#F97316" if score >= 60 else ("#EAB308" if score >= 40 else "#22C55E"))
    )
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 20, 'color': "#E2E8F0"}},
        number={'suffix': "/100", 'font': {'size': 32, 'color': gauge_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': gauge_color},
            'bgcolor': "#1E293B",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(34, 197, 94, 0.25)'},
                {'range': [40, 60], 'color': 'rgba(234, 179, 8, 0.25)'},
                {'range': [60, 75], 'color': 'rgba(249, 115, 22, 0.25)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
            ],
            'threshold': {
                'line': {'color': gauge_color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=20, t=50, b=20), height=260)
    return fig

def create_confidence_gauge(confidence_score, title="Confidence Score"):
    """
    Creates a Plotly gauge chart for Risk Assessment / Prediction Confidence (0-100%).
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score,
        title={'text': title, 'font': {'size': 20, 'color': "#E2E8F0"}},
        number={'suffix': "%", 'font': {'size': 32, 'color': "#3B82F6"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
            'bar': {'color': "#3B82F6"},
            'bgcolor': "#1E293B",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [50, 80], 'color': 'rgba(234, 179, 8, 0.2)'},
                {'range': [80, 100], 'color': 'rgba(59, 130, 246, 0.25)'}
            ]
        }
    ))
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=20, t=50, b=20), height=260)
    return fig

def create_feature_contribution_chart(breakdown_dict):
    """
    Creates a horizontal bar chart displaying factor contributions to Danger Score.
    """
    pct_dict = breakdown_dict["percentage_contributions"]
    df_contrib = pd.DataFrame({
        "Factor": list(pct_dict.keys()),
        "Contribution_Pct": list(pct_dict.values())
    }).sort_values(by="Contribution_Pct", ascending=True)

    fig = px.bar(
        df_contrib,
        x="Contribution_Pct",
        y="Factor",
        orientation='h',
        text="Contribution_Pct",
        color="Contribution_Pct",
        color_continuous_scale="Reds",
        title="Factor Contribution Breakdown to Danger Score"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=20, t=40, b=20), height=280, coloraxis_showscale=False)
    return fig

def create_model_feature_importance_chart(feat_imp_df):
    """
    Creates a horizontal bar chart of Random Forest feature importances.
    """
    df_imp = feat_imp_df.copy().sort_values(by="Importance_Pct", ascending=True)
    df_imp["Clean_Feature"] = df_imp["Feature"].str.replace("_", " ")

    fig = px.bar(
        df_imp,
        x="Importance_Pct",
        y="Clean_Feature",
        orientation='h',
        text="Importance_Pct",
        color="Importance_Pct",
        color_continuous_scale="Viridis",
        title="Random Forest Feature Importance Breakdown"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=40, t=40, b=20), height=320, coloraxis_showscale=False)
    return fig

def create_model_comparison_chart(comparison_df):
    """
    Creates a grouped bar chart comparing Validation MAE and Test R2 across models.
    """
    fig = px.bar(
        comparison_df,
        x="Model",
        y="CV_Val_MAE",
        color="Model",
        text="CV_Val_MAE",
        title="Model Comparison (5-Fold Cross-Validation MAE - Lower is Better)",
        color_discrete_sequence=["#38BDF8", "#818CF8", "#F472B6"]
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=20, t=40, b=20), height=300)
    return fig

def create_actual_vs_predicted_chart(actual_vs_pred_df):
    """
    Creates a Scatter Plot comparing Actual vs Predicted Danger Scores on test set.
    """
    fig = px.scatter(
        actual_vs_pred_df,
        x="Actual_Danger_Score",
        y="Predicted_Danger_Score",
        color="Absolute_Error",
        color_continuous_scale="Reds",
        title="Actual vs Predicted Danger Scores (Test Set Evaluation)",
        labels={"Actual_Danger_Score": "Actual Danger Score", "Predicted_Danger_Score": "Predicted Danger Score"}
    )
    # Add ideal reference line y=x
    max_val = max(actual_vs_pred_df["Actual_Danger_Score"].max(), actual_vs_pred_df["Predicted_Danger_Score"].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        name='Perfect Prediction (y = x)',
        line=dict(color='#22C55E', dash='dash', width=2)
    ))
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=20, t=40, b=20), height=320)
    return fig

def create_crime_category_pie(row):
    """
    Creates a Donut chart of crime categories for a single area.
    """
    category_map = {
        "Rape": "Rape",
        "Kidnapping_and_Abduction": "Kidnapping & Abduction",
        "Dowry_Deaths": "Dowry Deaths",
        "Assault_on_Women": "Assault on Women",
        "Cruelty_by_Husband_or_Relatives": "Cruelty by Husband/Relatives",
        "Sexual_Harassment": "Sexual Harassment",
        "Cyber_Crime": "Cyber Crime",
        "Other_Crimes_Against_Women": "Other Crimes"
    }
    
    labels = []
    values = []
    for col, label in category_map.items():
        if col in row and row[col] > 0:
            labels.append(label)
            values.append(row[col])

    if not values:
        labels = ["No Data"]
        values = [1]

    fig = px.pie(
        names=labels,
        values=values,
        hole=0.45,
        title=f"Crime Category Breakdown - {row['City']} ({row['Year']})"
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#0F172A', width=2)))
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=20, t=40, b=20), height=320)
    return fig

def create_area_comparison_chart(df, current_year=2022):
    """
    Creates a horizontal bar chart comparing Danger Scores across districts.
    """
    df_year = df[df["Year"] == current_year].sort_values(by="Danger_Score", ascending=True).tail(20)
    
    fig = px.bar(
        df_year,
        x="Danger_Score",
        y="City",
        orientation='h',
        color="Danger_Score",
        color_continuous_scale="Reds",
        title=f"Top 20 District Risk Ranking - Danger Scores ({current_year})",
        text="Danger_Score"
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(**DARK_THEME_LAYOUT, margin=dict(l=20, r=40, t=40, b=20), height=480, coloraxis_showscale=False)
    return fig

def create_year_trend_chart(df, selected_city):
    """
    Creates a line chart showing multi-year crime trend for a selected district.
    """
    df_city = df[df["City"] == selected_city].sort_values(by="Year")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_city["Year"].astype(str),
        y=df_city["Total_Crime_Against_Women"],
        mode='lines+markers+text',
        name='Total Reported Crimes',
        line=dict(color='#3B82F6', width=3),
        marker=dict(size=10, color='#60A5FA'),
        text=df_city["Total_Crime_Against_Women"],
        textposition="top center"
    ))
    
    fig.update_layout(
        **DARK_THEME_LAYOUT,
        title=f"Multi-Year Reported Crime Trend ({df['Year'].min()}–{df['Year'].max()}) - {selected_city}",
        xaxis_title="Year",
        yaxis_title="Total Reported Crimes Against Women",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    return fig

def build_interactive_folium_map(df, selected_year=2022):
    """
    Builds an interactive Folium map displaying district risk markers with rich popups.
    """
    df_year = df[df["Year"] == selected_year]
    
    # Center map on India
    map_center = [20.5937, 78.9629]
    m = folium.Map(location=map_center, zoom_start=5, tiles="OpenStreetMap")
    
    for idx, row in df_year.iterrows():
        lat = row.get("Latitude", None)
        lon = row.get("Longitude", None)
        if pd.isna(lat) or pd.isna(lon):
            continue
            
        danger = row.get("Danger_Score", 50.0)
        confidence = row.get("Confidence_Score", 85.0)
        total_crime = int(row.get("Total_Crime_Against_Women", 0))
        cluster_id = row.get("Cluster", -1)
        cluster_name = f"Cluster {cluster_id + 1}" if cluster_id != -1 else "Noise (Outlier)"
        anomaly_status = row.get("Anomaly_Status", "Normal")
        
        # Color coding marker
        if danger >= 75:
            color = "#EF4444" # Red
        elif danger >= 60:
            color = "#F97316" # Orange
        elif danger >= 40:
            color = "#EAB308" # Yellow
        else:
            color = "#22C55E" # Green

        # Popup HTML Content
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 230px; color: #0F172A;">
            <h4 style="margin: 0 0 5px 0; color: #1E293B; border-bottom: 2px solid {color}; padding-bottom: 4px;">
                📍 {row['City']}, {row['State']}
            </h4>
            <p style="margin: 4px 0;"><b>Danger Score:</b> <span style="font-size: 16px; color: {color}; font-weight: bold;">{danger}/100</span></p>
            <p style="margin: 4px 0;"><b>Risk Confidence:</b> <span style="color: #2563EB; font-weight: bold;">{confidence}%</span></p>
            <p style="margin: 4px 0;"><b>Reported Crimes:</b> {total_crime:,}</p>
            <p style="margin: 4px 0;"><b>DBSCAN Group:</b> {cluster_name}</p>
            <p style="margin: 4px 0;"><b>Anomaly Status:</b> <span style="color: {'#DC2626' if anomaly_status == 'Anomalous' else '#16A34A'}; font-weight: bold;">{anomaly_status}</span></p>
            <hr style="margin: 8px 0; border: 0; border-top: 1px solid #E2E8F0;">
            <p style="margin: 0; font-size: 11px; color: #64748B;">Source: NCRB Crime in India {selected_year}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=7 + (danger / 12),
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['City']}: Danger Score {danger}/100",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=2
        ).add_to(m)

    return m
