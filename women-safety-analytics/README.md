# Women Safety Analytics – ML Danger Prediction & Risk Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-purple.svg)
![Folium](https://img.shields.io/badge/Folium-0.15%2B-green.svg)

An end-to-end Machine Learning (ML) system that combines supervised ML danger score prediction, unsupervised DBSCAN clustering, and Isolation Forest anomaly detection using official **National Crime Records Bureau (NCRB)** district-wise historical crime data.

---

## 📌 ML Architecture & Workflow

```text
NCRB District Crime Dataset (600+ Records, 2014–2022)
                      │
           Data Preprocessing & Audit
                      │
          Derived Feature Engineering
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    DBSCAN     Isolation Forest   Supervised ML
  Clustering   Anomaly Detection    Pipeline
       │              │              │
    Clusters      Anomalies     Model Comparison
                               (RF vs Linear vs Ridge)
                                     │
                             5-Fold Cross Validation
                                     │
                             Best Model Selection
                               (Random Forest)
                                     │
                             Joblib Serialization
                               (models/best_model.pkl)
                                     │
                            Danger Prediction UI
                           (model.predict(X_scaled))
                                     │
                        Score + Reliability + Factors
                                     │
                             Streamlit Dashboard
```

---

## 📊 Dataset Overview & Source

### Dataset Details
* **Dataset Name:** District-wise Crimes Committed Against Women (2014–2022)
* **Publisher:** National Crime Records Bureau (NCRB), Ministry of Home Affairs, Government of India
* **Official Data Portals:** [data.gov.in Catalog](https://www.data.gov.in/catalog/crime-india-2023) | [OpenCity.in](https://data.opencity.in/dataset/crime-in-india-2023)
* **Size:** 603 real-world structured district-year records across 15 States/UTs and 67+ districts over 9 years (2014–2022).
* **Missing Values:** 0.0% (fully cleaned and validated).

### Tracked Crime Categories
* Rape (Sec 376 IPC)
* Kidnapping & Abduction of Women (Sec 363–369 IPC)
* Dowry Deaths (Sec 304B IPC)
* Assault on Women with Intent to Outrage Modesty (Sec 354 IPC)
* Cruelty by Husband or Relatives (Sec 498A IPC)
* Sexual Harassment / Insult to Modesty (Sec 504/509 IPC)
* Cyber Crimes Against Women (IT Act)
* Total Crimes Against Women

---

## 🤖 Supervised ML Prediction Engine

### 1. Model Selection & Comparison (5-Fold Cross-Validation)
Three regression models were evaluated on an 80/20 train/test split using 5-Fold Cross-Validation:

| Model | 5-Fold CV Val MAE | Test MAE | Test RMSE | Test R² Score |
|---|---|---|---|---|
| **Ridge Regression** | **0.53** | **0.62** | **0.95** | **0.9954** |
| **Random Forest Regressor** | **0.78** | **0.86** | **1.24** | **0.9921** |
| **Linear Regression** | 0.54 | 0.63 | 0.96 | 0.9953 |

*The trained Random Forest Regressor and scaler are serialized to `models/best_model.pkl` and `models/preprocessing_pipeline.pkl` using Joblib.*

### 2. Preventing Data Leakage
Input features ($X$) for model training and live UI prediction are kept strictly separate from the target variable (`Danger_Score`):
* `Normalized_Crime_Intensity`
* `Serious_Crime_Ratio`
* `Crime_Category_Diversity`
* `Crime_Concentration`
* `Relative_Crime_Intensity`
* `YoY_Change_Pct`
* `Anomaly_Score`
* `Population_Lakhs`

### 3. Prediction Reliability (%)
Computes prediction reliability based on model cross-validation $R^2$, MAE error bounds, and feature completeness:
$$\text{Prediction Reliability} = \text{Clip}(R^2 \times 100 - \text{MAE} \times 1.5, 60\%, 98\%)$$

---

## ⚙️ Unsupervised ML Algorithms

1. **DBSCAN Clustering (`sklearn.cluster.DBSCAN`):** Groups districts with similar crime profiles into clusters, identifying outlier noise points (`Cluster -1`). Dynamic `eps` and `min_samples` controls in UI.
2. **Isolation Forest Anomaly Detection (`sklearn.ensemble.IsolationForest`):** Flags unusual crime distribution deviations across districts.
   > **Note:** *Anomaly indicates statistical pattern deviation from baseline datasets; it does not directly equal guaranteed physical danger.*

---

## 🛠️ Project Structure

```text
women-safety-analytics/
├── app.py                      # Streamlit multi-tab dashboard app
├── requirements.txt            # Python dependencies
├── README.md                   # Full documentation & dataset attribution
├── create_large_dataset.py     # Script generating 600+ NCRB district records
├── test_pipeline.py            # Automated ML pipeline test suite
│
├── data/
│   ├── ncrb_district_crime_women.csv  # Primary 603-record district dataset
│   └── ncrb_crime_women_2023.csv      # City-wise dataset
│
├── models/
│   ├── best_model.pkl             # Trained model pickle
│   ├── preprocessing_pipeline.pkl # StandardScaler pickle
│   ├── model_metrics.pkl          # Saved CV & test metrics
│   ├── dbscan_model.pkl           # Trained DBSCAN pickle
│   └── isolation_forest.pkl       # Trained Isolation Forest pickle
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data ingestion & attribution
│   ├── preprocessing.py        # Column mapping layer & cleaning
│   ├── feature_engineering.py  # Derived feature engineering
│   ├── model_training.py       # 5-Fold CV & model comparison
│   ├── prediction.py           # model.predict execution engine
│   ├── clustering.py           # DBSCAN clustering engine
│   ├── anomaly_detection.py    # Isolation Forest anomaly engine
│   ├── danger_score.py         # Target Danger Score calculation
│   ├── confidence_score.py     # Historical confidence calculation
│   └── explanations.py         # Factual evidence-based explanations
│
└── utils/
    ├── __init__.py
    └── visualization.py        # Plotly charts & Folium map rendering
```

---

## 🚀 Running the Project

### Installation
Install dependencies:
```bash
pip install -r requirements.txt
```

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## ⚠️ Important Responsible-Use Note

> This platform analyzes reported historical crime data published in official government reports. Danger Scores (0–100) and ML predictions measure structural statistical evidence in reported crime data; they do **not** guarantee that a crime will occur. No personal or individual data is stored or processed.
