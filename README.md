# 🏥 Hospital Patient Profiling

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Hospital--Patient-181717.svg?logo=github)](https://github.com/durgaparamashivam2007-glitch/Hospital-Patient)

> **Grouping ICU Patients Based on Continuous Clinical Vitals & Measurements**  
> *A Production-Ready Unsupervised Machine Learning Application Built with Scikit-Learn, Plotly, and Streamlit*

---

## 📋 Table of Contents
- [📌 Project Overview](#-project-overview)
- [🎯 Key Objectives](#-key-objectives)
- [📊 Dataset Specifications](#-dataset-specifications)
- [🛠️ Machine Learning Architecture](#%EF%B8%8F-machine-learning-architecture)
- [🖥️ Streamlit Web Dashboard Features](#%EF%B8%8F-streamlit-web-dashboard-features)
- [📁 Project Directory Structure](#-project-directory-structure)
- [💻 Local Setup & Installation](#-local-setup--installation)
- [☁️ Streamlit Community Cloud Deployment](#%EF%B8%8F-streamlit-community-cloud-deployment)
- [⚠️ Medical Disclaimer](#%EF%B8%8F-medical-disclaimer)

---

## 📌 Project Overview
**Hospital Patient Profiling** is a complete, interactive, end-to-end unsupervised machine learning solution designed to group Intensive Care Unit (ICU) patients into clinically distinct, statistically sound patient profiles.

Using **K-Means Clustering** applied to continuous physiological parameters (vital signs, lab indicators, and severity metrics), the application reveals distinct multi-system patient profiles without relying on clinical outcome targets during model training.

The accompanying multi-page **Streamlit Web Application** offers clinicians and healthcare data analysts an intuitive, visual environment to:
- Interactively explore patient clusters across clinical features.
- Evaluate cluster quality using the **Elbow Method** (Inertia) and **Silhouette Scores**.
- Visualize high-dimensional patient vitals in 2D via **Principal Component Analysis (PCA)**.
- Classify new incoming patient vitals into trained profiles in real-time.

---

## 🎯 Key Objectives
1. **Unsupervised Patient Segmentation**: Partition 4,580 ICU patient records into homogeneous, interpretable clusters using K-Means clustering.
2. **Data Leakage Elimination**: Strict separation of predictive features from patient identifiers (`patient_id`) and post-hoc outcome variables (`survived_icu`). ICU survival is evaluated purely as a descriptive statistic.
3. **Rigorous Optimal Cluster Selection**: Interactive algorithmic evaluation ($K = 2 \dots 10$) using Inertia and Silhouette coefficient analysis.
4. **Interactive Clinical Profiler**: Instant feature z-score profiling, automated neutral profile labeling, feature distribution overlays, and real-time patient assignment.

---

## 📊 Dataset Specifications
The model utilizes the Kaggle ICU Patient Cohort (`train.csv`).

- **Total Patient Records**: 4,580 rows
- **Total Clinical Columns**: 16 columns

### Continuous & Categorical Features Used for Clustering (14 Features):
| Feature Name | Clinical Description | Data Type | Imputation Strategy |
| :--- | :--- | :--- | :--- |
| `age` | Patient age in years | Integer | None |
| `gender` | Binary gender indicator (0=Female, 1=Male) | Integer | Most Frequent |
| `bmi` | Body Mass Index ($\text{kg/m}^2$) | Float | Median Imputation |
| `heart_rate` | Resting heart rate (beats per minute) | Float | None |
| `systolic_bp` | Systolic Blood Pressure (mmHg) | Float | None |
| `diastolic_bp` | Diastolic Blood Pressure (mmHg) | Float | None |
| `respiratory_rate` | Respiratory rate (breaths per minute) | Float | Median Imputation |
| `oxygen_saturation` | Blood oxygen saturation level ($\text{SpO}_2\%$) | Float | Median Imputation |
| `glucose` | Serum glucose level (mg/dL) | Float | None |
| `creatinine` | Kidney marker / Serum Creatinine (mg/dL) | Float | Median Imputation |
| `wbc_count` | White Blood Cell Count ($\text{K/\mu L}$) | Float | Median Imputation |
| `severity_score` | Composite clinical severity score | Integer | None |
| `ventilation_required` | Mechanical ventilation indicator (0/1) | Integer | None |
| `chronic_conditions` | Pre-existing chronic condition count | Integer | None |

### Excluded Columns:
- `patient_id`: Non-clinical patient identifier.
- `survived_icu`: ICU survival outcome flag (0 = Deceased, 1 = Survived). *Excluded to prevent data leakage during unsupervised clustering.*

---

## 🛠️ Machine Learning Architecture

```
                      ┌──────────────────────────────┐
                      │    Kaggle ICU Dataset        │
                      │        (train.csv)           │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ Missing Value Imputation     │
                      │ (Median & Categorical)       │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ Feature Standard Scaling     │
                      │ (StandardScaler)             │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │ K-Means Clustering           │
                      │ (n_clusters = K, K=2..10)    │
                      └──────────────┬───────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Optimal K        │       │ 2D PCA           │       │ Real-Time        │
│ Analysis         │       │ Projection       │       │ Patient          │
│ (Elbow &         │       │ (Scatter Plot    │       │ Profile          │
│ Silhouette)      │       │ Visualization)   │       │ Prediction       │
└──────────────────┘       └──────────────────┘       └──────────────────┘
```

### Module Breakdown:
1. **`src/data_preprocessing.py`**: Handles missing value imputation via `SimpleImputer` and feature standardization via `StandardScaler`.
2. **`src/clustering.py`**: Trains `KMeans(n_clusters=K, random_state=42)` and computes cluster metrics across $K=2\dots 10$.
3. **`src/visualization.py`**: Generates interactive Plotly visualizations (Elbow curve, Silhouette plot, 2D PCA scatter plots, cluster profile heatmaps, survival distribution charts).
4. **`src/profiling.py`**: Calculates population z-scores, auto-assigns evidence-based clinical descriptions, and classifies individual new patient entries.
5. **`src/utils.py`**: Central dataset validation, file resolution, and configuration constants.

---

## 🖥️ Streamlit Web Dashboard Features

The web interface is organized into intuitive navigation tabs:

- 📊 **Dataset Overview**: Detailed summary statistics, missing value distribution, and correlation heatmaps.
- 🎯 **Clustering & Optimal K Analysis**: Interactive Elbow Plot and Silhouette Analysis for selecting $K$.
- 🌌 **2D PCA Visualization**: High-dimensional cluster separation visualizer using Principal Component Analysis.
- 🧬 **Cluster Profiles**: Comparative heatmaps, z-score profiles, and post-hoc ICU survival rates.
- 👤 **New Patient Profiling**: Dynamic input controls allowing clinicians to assign real-time patient vitals to trained profiles with confidence distances.

---

## 📁 Project Directory Structure

```
Hospital-Patient/
├── app.py                      # Streamlit Multi-Page Application Entry Point
├── requirements.txt            # Python Dependencies & Versions
├── README.md                   # Complete Technical Documentation
├── .gitignore                  # Git Ignore Rules
│
├── data/                       # Dataset Storage
│   ├── train.csv               # Primary ICU Patient Dataset
│   ├── test.csv                # Test Patient Dataset
│   └── sample_submission.csv   # Sample Submission Format
│
├── models/                     # Saved Model Artifacts & Pipelines
│   ├── .gitkeep
│   ├── clustering_model.joblib # Trained KMeans Model
│   ├── preprocessing_pipeline.joblib
│   ├── cluster_profiles.joblib
│   └── feature_config.joblib
│
├── src/                        # Modular Source Code
│   ├── __init__.py
│   ├── data_preprocessing.py   # Sklearn Preprocessing Pipelines
│   ├── clustering.py           # KMeans & Evaluation Engine
│   ├── profiling.py            # Statistical Profiling & Prediction
│   ├── visualization.py        # Plotly Interactivity Engine
│   └── utils.py                # Dataset Validation & Helper Functions
│
└── notebooks/                  # Interactive Analysis Notebooks
    └── patient_profiling.ipynb # End-to-End Jupyter Analysis Notebook
```

---

## 💻 Local Setup & Installation

### Prerequisites
- **Python 3.9+** installed.
- **Git** installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/durgaparamashivam2007-glitch/Hospital-Patient.git
cd Hospital-Patient
```

### Step 2: Create & Activate Virtual Environment (Optional but Recommended)
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Web Application
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## ☁️ Streamlit Community Cloud Deployment

To host this project live for free on **Streamlit Community Cloud**:

1. Push all latest changes to GitHub:
   ```bash
   git push -u origin main
   ```
2. Navigate to [Streamlit Community Cloud](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **New App**.
4. Select the repository: `durgaparamashivam2007-glitch/Hospital-Patient`.
5. Set **Main file path** to `app.py`.
6. Click **Deploy!**.

---

## ⚠️ Medical Disclaimer
> **IMPORTANT NOTICE:**  
> *This application and its generated patient profiles are intended solely for educational, analytical, and research purposes. The statistical clusters produced by unsupervised machine learning do NOT constitute clinical diagnosis, prognosis, or medical advice. Healthcare professionals must make all clinical decisions independently using verified medical standards.*
