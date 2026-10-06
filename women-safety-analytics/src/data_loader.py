import os
import pandas as pd
import streamlit as st

DATA_ATTRIBUTION = {
    "source_name": "National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India",
    "catalog_title": "Crime in India 2023 - Crimes Against Women (City-Wise)",
    "official_url": "https://www.data.gov.in/catalog/crime-india-2023",
    "resource_url": "https://data.opencity.in/dataset/crime-in-india-2023",
    "disclaimer": "This application utilizes historical aggregated crime data published in official NCRB reports. Data granularity is at the metropolitan city level."
}

def load_crime_data(filepath=None):
    """
    Loads NCRB crime dataset or fallback sample dataset.
    """
    if filepath is None or not os.path.exists(filepath):
        # Default path relative to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, "data", "ncrb_crime_women_2023.csv")
        
        if not os.path.exists(filepath):
            # Fallback to local cwd
            filepath = os.path.join("data", "ncrb_crime_women_2023.csv")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = pd.read_csv(filepath)
    return df, DATA_ATTRIBUTION

def get_data_metadata(filepath):
    """
    Returns dataset metadata tag distinguishing real NCRB data from sample data.
    """
    filename = os.path.basename(filepath)
    if "ncrb" in filename.lower():
        return {
            "type": "Real NCRB Data",
            "badge_color": "green",
            "description": "Official National Crime Records Bureau (NCRB) Crime in India 2021–2023 Metropolitan City Data"
        }
    else:
        return {
            "type": "Derived Sample Data",
            "badge_color": "orange",
            "description": "Sample/District Level Derived Crime Data for Analysis Demonstration"
        }
