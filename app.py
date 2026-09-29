import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Add current directory and src/ to sys.path to support execution from any directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import (
    load_dataset,
    validate_dataset,
    DEFAULT_CLINICAL_FEATURES,
    EXCLUDED_FEATURES,
    FEATURE_LABELS,
    ensure_directories
)
from src.data_preprocessing import (
    preprocess_dataset,
    inspect_preprocessing_summary
)
from src.clustering import (
    train_kmeans,
    evaluate_k_range,
    compute_pca_2d
)
from src.profiling import (
    compute_cluster_profiles,
    predict_patient_profile,
    save_model_artifacts,
    load_model_artifacts
)
from src.visualization import (
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_cluster_distribution,
    plot_pca_2d,
    plot_profile_heatmap,
    plot_feature_comparisons,
    plot_survival_by_cluster,
    CLUSTER_COLORS
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Clean Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hospital Patient Profiling",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern professional aesthetic
st.markdown("""
    <style>
    /* Main layout & background styling */
    .main {
        background-color: #F8F9FA;
    }
    .stAppHeader {
        background-color: transparent;
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E293B;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #64748B;
        font-weight: 600;
    }
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    /* Headers & Subtitles */
    .app-header {
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    .app-header p {
        margin: 4px 0 0 0;
        font-size: 1.1rem;
        color: #94A3B8;
    }
    .disclaimer-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 20px;
        font-size: 0.88rem;
        color: #92400E;
    }
    .badge-profile {
        background-color: #0284C7;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Ensure required directories exist
ensure_directories()

# -----------------------------------------------------------------------------
# Cached Data Loading & Model Training Functions
# -----------------------------------------------------------------------------
@st.cache_data
def get_cached_data():
    df, err = load_dataset()
    return df, err

@st.cache_data
def get_k_evaluation(df_data, feature_cols):
    X_scaled, pipeline, df_imputed, _ = preprocess_dataset(df_data, feature_cols)
    k_df, optimal_k = evaluate_k_range(X_scaled)
    return k_df, optimal_k

# Load dataset
df, load_error = get_cached_data()

# Header display
st.markdown("""
    <div class="app-header">
        <h1>🏥 Hospital Patient Profiling</h1>
        <p>Grouping Patients Based on Clinical Measurements | Unsupervised Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ **Dataset Error**: {load_error}")
    st.info("Please verify that `train.csv` is present in the `data/` folder.")
    st.stop()

# -----------------------------------------------------------------------------
# Sidebar Navigation Menu
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/hospital-2.png", width=70)
st.sidebar.title("Navigation Menu")

pages = [
    "1. Dashboard",
    "2. Dataset Explorer",
    "3. Data Preprocessing",
    "4. Clustering Analysis",
    "5. Patient Profiles",
    "6. Profile a New Patient",
    "7. About Project"
]

selected_page = st.sidebar.radio("Select Application Module", pages)

# Sidebar controls for default features
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Configuration")
selected_k = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4, step=1)

feature_cols = DEFAULT_CLINICAL_FEATURES

# Pipeline & Model Training step
X_scaled, pipeline, df_imputed, feature_names = preprocess_dataset(df, feature_cols)
kmeans_model, labels, inertia, sil_score = train_kmeans(X_scaled, n_clusters=selected_k)
profiles_df, pop_stats = compute_cluster_profiles(df_imputed, labels, feature_cols)
pca_df, pca_obj = compute_pca_2d(X_scaled)

# Save artifacts for reusability
save_model_artifacts(kmeans_model, pipeline, profiles_df, feature_cols)


# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if selected_page == "1. Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown("Overview of ICU patient population clusters and clinical measurement metrics.")

    # KPI Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total ICU Patients", f"{len(df):,}")
    with c2:
        st.metric("Active Patient Profiles", f"{selected_k}")
    with c3:
        st.metric("Selected K-Means K", f"{selected_k}")
    with c4:
        st.metric("Silhouette Score", f"{sil_score:.3f}")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.plotly_chart(plot_cluster_distribution(profiles_df), use_container_width=True)
    with col_right:
        st.plotly_chart(plot_pca_2d(pca_df, labels, df), use_container_width=True)

    st.markdown("### 📋 Cluster Profile Summary Overview")
    display_summary_cols = ['profile_name', 'auto_description', 'patient_count', 'patient_percentage', 'avg_age', 'avg_glucose', 'avg_systolic_bp', 'avg_oxygen_saturation', 'avg_severity_score', 'icu_survival_pct']
    summary_rename = {
        'profile_name': 'Profile',
        'auto_description': 'Description',
        'patient_count': 'Patients',
        'patient_percentage': '% Population',
        'avg_age': 'Avg Age',
        'avg_glucose': 'Avg Glucose',
        'avg_systolic_bp': 'Avg Sys BP',
        'avg_oxygen_saturation': 'Avg O2 Sat (%)',
        'avg_severity_score': 'Avg Severity',
        'icu_survival_pct': 'ICU Survival %'
    }
    st.dataframe(
        profiles_df[display_summary_cols].rename(columns=summary_rename),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
        <div class="disclaimer-box">
            <b>⚠️ Medical Disclaimer:</b> This application is intended for educational and data-analysis purposes only. Patient profiles are generated using statistical clustering of clinical measurements and should not be interpreted as medical diagnoses, treatment recommendations, or clinical decisions.
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DATASET EXPLORER
# =============================================================================
elif selected_page == "2. Dataset Explorer":
    st.title("🔍 Kaggle ICU Dataset Explorer")
    st.markdown("Inspect, search, and filter the raw patient records from `train.csv`.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{df.shape[0]:,}")
    col2.metric("Total Columns", f"{df.shape[1]}")
    col3.metric("Numeric Features", f"{len(df.select_dtypes(include=[np.number]).columns)}")
    col4.metric("Total Missing Cells", f"{df.isnull().sum().sum():,}")
    col5.metric("Duplicate Rows", f"{df.duplicated().sum()}")

    st.markdown("---")
    st.subheader("Filter Patient Records")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        age_min, age_max = int(df['age'].min()), int(df['age'].max())
        selected_age = st.slider("Age Range", age_min, age_max, (age_min, age_max))
    with f2:
        gender_options = ["All", "0 (Female)", "1 (Male)"]
        selected_gender = st.selectbox("Gender Filter", gender_options)
    with f3:
        search_id = st.text_input("Search Patient ID", placeholder="e.g. ICU_00301")

    # Apply filters
    filtered_df = df[(df['age'] >= selected_age[0]) & (df['age'] <= selected_age[1])]
    if selected_gender != "All":
        g_val = 0 if "0" in selected_gender else 1
        filtered_df = filtered_df[filtered_df['gender'] == g_val]
    if search_id.strip():
        filtered_df = filtered_df[filtered_df['patient_id'].str.contains(search_id.strip(), case=False, na=False)]

    st.markdown(f"Displaying **{len(filtered_df):,}** matching patient records:")
    st.dataframe(filtered_df, use_container_width=True)

    with st.expander("📊 Column Data Types & Missing Values Breakdown"):
        dtypes_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 3: DATA PREPROCESSING
# =============================================================================
elif selected_page == "3. Data Preprocessing":
    st.title("⚙️ Data Preprocessing Pipeline")
    st.markdown("Inspect raw data issues, missing value imputations, encoding, and scaling transformers.")

    summary = inspect_preprocessing_summary(df, feature_cols)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Dataset Rows", f"{summary['total_rows']:,}")
    c2.metric("Clustering Features", f"{summary['feature_count']}")
    c3.metric("Missing Values Imputed", f"{summary['missing_total']:,}")
    c4.metric("Duplicate Rows Found", f"{summary['duplicate_rows']}")

    st.markdown("---")
    st.subheader("1. Missing Values Before vs After Imputation")

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Missing Values in Raw `train.csv`**")
        missing_series = df[feature_cols].isnull().sum()
        missing_df = pd.DataFrame({
            'Feature': missing_series.index,
            'Missing Count': missing_series.values,
            'Status': ['⚠️ Needs Imputation' if v > 0 else '✅ Complete' for v in missing_series.values]
        })
        st.dataframe(missing_df, use_container_width=True, hide_index=True)

    with col_after:
        st.markdown("**Processed Feature Matrix Status**")
        processed_missing = pd.DataFrame(X_scaled).isnull().sum().sum()
        st.success(f"✅ Zero missing values remaining after Scikit-Learn Pipeline imputation.")
        st.info(f"Shape of Processed Feature Matrix `X_scaled`: **{X_scaled.shape}**")

    st.markdown("---")
    st.subheader("2. Preprocessing Architecture Breakdown")
    st.markdown("""
    The preprocessing pipeline uses scikit-learn `Pipeline` and `ColumnTransformer`:
    - **Numerical Features** (`bmi`, `respiratory_rate`, `oxygen_saturation`, `creatinine`, `wbc_count`, etc.):
        1. **Median Imputation** (`SimpleImputer(strategy='median')`): Replaces missing clinical values with column medians.
        2. **Standard Scaling** (`StandardScaler()`): Standardizes features to mean=0, std=1 for distance-based K-Means.
    - **Categorical Features** (`gender`):
        1. **Most Frequent Imputation** (`SimpleImputer(strategy='most_frequent')`).
        2. **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore')`).
    - **Excluded Columns**:
        - `patient_id`: Identifier (non-clinical).
        - `survived_icu`: ICU outcome variable (preventing data leakage).
    """)

    with st.expander("🔍 Preview Standardized Feature Matrix Sample (X_scaled)"):
        scaled_df_preview = pd.DataFrame(X_scaled, columns=feature_names).head(10)
        st.dataframe(scaled_df_preview, use_container_width=True)


# =============================================================================
# PAGE 4: CLUSTERING ANALYSIS
# =============================================================================
elif selected_page == "4. Clustering Analysis":
    st.title("📈 K-Means Clustering & Optimal K Analysis")
    st.markdown("Analyze K-Means clustering performance using the Elbow Method and Silhouette Analysis.")

    st.subheader("Optimal K Determination")
    auto_opt = st.checkbox("Automatically calculate metrics for K = 2 to 10", value=True)

    if auto_opt:
        with st.spinner("Calculating Inertia & Silhouette Scores across K=2..10..."):
            k_df, optimal_k = get_k_evaluation(df, feature_cols)

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            st.success(f"💡 **Suggested Optimal K**: **{optimal_k}** (Based on Highest Silhouette Score)")
        with col_opt2:
            max_sil = k_df[k_df['k'] == optimal_k]['silhouette_score'].iloc[0]
            st.info(f"Highest Silhouette Score: **{max_sil:.3f}**")

        c_elbow, c_sil = st.columns(2)
        with c_elbow:
            st.plotly_chart(plot_elbow_curve(k_df, optimal_k), use_container_width=True)
        with c_sil:
            st.plotly_chart(plot_silhouette_scores(k_df, optimal_k), use_container_width=True)

    st.markdown("---")
    st.subheader(f"Current Model Cluster Visualizations (K = {selected_k})")

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.plotly_chart(plot_pca_2d(pca_df, labels, df), use_container_width=True)
    with col_vis2:
        st.plotly_chart(plot_profile_heatmap(profiles_df, feature_cols), use_container_width=True)


# =============================================================================
# PAGE 5: PATIENT PROFILES
# =============================================================================
elif selected_page == "5. Patient Profiles":
    st.title("👤 Patient Profiles & Clinical Statistical Comparison")
    st.markdown("Detailed breakdown of statistical clinical characteristics for each generated patient group.")

    st.subheader("📊 Patient Cluster Profile Table")
    
    prof_cols = ['profile_name', 'auto_description', 'patient_count', 'patient_percentage', 'avg_age', 'avg_bmi', 'avg_heart_rate', 'avg_systolic_bp', 'avg_diastolic_bp', 'avg_oxygen_saturation', 'avg_glucose', 'avg_creatinine', 'avg_wbc_count', 'avg_severity_score', 'ventilation_pct', 'icu_survival_pct']
    prof_rename = {
        'profile_name': 'Profile',
        'auto_description': 'Auto Description',
        'patient_count': 'Patients',
        'patient_percentage': '% Total',
        'avg_age': 'Age',
        'avg_bmi': 'BMI',
        'avg_heart_rate': 'Heart Rate',
        'avg_systolic_bp': 'Sys BP',
        'avg_diastolic_bp': 'Dia BP',
        'avg_oxygen_saturation': 'O2 Sat (%)',
        'avg_glucose': 'Glucose',
        'avg_creatinine': 'Creatinine',
        'avg_wbc_count': 'WBC',
        'avg_severity_score': 'Severity',
        'ventilation_pct': 'Ventilation %',
        'icu_survival_pct': 'ICU Survival %'
    }
    
    st.dataframe(
        profiles_df[prof_cols].rename(columns=prof_rename),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(plot_feature_comparisons(profiles_df, feature_cols), use_container_width=True)
    with col_chart2:
        survival_fig = plot_survival_by_cluster(profiles_df)
        if survival_fig:
            st.plotly_chart(survival_fig, use_container_width=True)
            st.caption("ℹ️ *ICU survival is shown as a post-clustering descriptive outcome and was not used to create the clusters.*")

    st.markdown("---")
    st.subheader("📋 Automatic Profile Descriptions & Key Characteristics")
    for _, r in profiles_df.iterrows():
        with st.expander(f"🔹 **{r['profile_name']}**: {r['auto_description']} ({r['patient_count']} patients, {r['patient_percentage']}%)"):
            st.markdown(f"""
            - **Demographics**: Avg Age {r['avg_age']} years | Avg BMI {r['avg_bmi']} kg/m²
            - **Hemodynamics**: Avg Heart Rate {r['avg_heart_rate']} bpm | Blood Pressure {r['avg_systolic_bp']}/{r['avg_diastolic_bp']} mmHg
            - **Metabolic & Labs**: Avg Glucose {r['avg_glucose']} mg/dL | Creatinine {r['avg_creatinine']} mg/dL | WBC {r['avg_wbc_count']} K/µL
            - **Respiratory & Severity**: Oxygen Saturation {r['avg_oxygen_saturation']}% | Severity Score {r['avg_severity_score']}
            - **Interventions & Outcomes**: Ventilation Required {r['ventilation_pct']}% | ICU Survival Rate {r['icu_survival_pct']}%
            """)


# =============================================================================
# PAGE 6: PROFILE A NEW PATIENT
# =============================================================================
elif selected_page == "6. Profile a New Patient":
    st.title("🩺 Profile a New Patient")
    st.markdown("Input clinical measurements for a new patient to determine their matching clinical profile.")

    st.info("Enter patient clinical measurements below (Patient ID and ICU Survival outcome are excluded to prevent data leakage).")

    with st.form("new_patient_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=60)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "0 (Female)" if x == 0 else "1 (Male)")
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=200.0, value=85.0, step=1.0)
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70.0, max_value=220.0, value=125.0, step=1.0)

        with col2:
            st.subheader("Respiratory & Labs")
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=140.0, value=80.0, step=1.0)
            respiratory_rate = st.number_input("Respiratory Rate (bpm)", min_value=8.0, max_value=50.0, value=20.0, step=1.0)
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=70.0, max_value=100.0, value=96.0, step=0.5)
            glucose = st.number_input("Glucose (mg/dL)", min_value=50.0, max_value=500.0, value=140.0, step=1.0)
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.2, max_value=15.0, value=1.2, step=0.1)

        with col3:
            st.subheader("Clinical Severity & History")
            wbc_count = st.number_input("WBC Count (K/µL)", min_value=1.0, max_value=40.0, value=11.0, step=0.5)
            severity_score = st.number_input("Severity Score", min_value=0, max_value=30, value=12)
            ventilation_required = st.selectbox("Ventilation Required", [0, 1], format_func=lambda x: "0 (No)" if x == 0 else "1 (Yes)")
            chronic_conditions = st.number_input("Chronic Conditions Count", min_value=0, max_value=10, value=1)

        submitted = st.form_submit_button("🩺 Assign Patient Profile")

    if submitted:
        patient_dict = {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'heart_rate': heart_rate,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'respiratory_rate': respiratory_rate,
            'oxygen_saturation': oxygen_saturation,
            'glucose': glucose,
            'creatinine': creatinine,
            'wbc_count': wbc_count,
            'severity_score': severity_score,
            'ventilation_required': ventilation_required,
            'chronic_conditions': chronic_conditions
        }

        cluster_id, profile_dict, comp_df = predict_patient_profile(
            patient_dict, pipeline, kmeans_model, profiles_df, feature_cols, pop_stats
        )

        st.markdown("---")
        st.subheader("🎯 Assignment Result")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
                <div style="background-color:#E0F2FE; border-left: 5px solid #0284C7; padding: 20px; border-radius: 8px;">
                    <h3 style="margin:0; color:#0369A1;">Assigned Profile:</h3>
                    <h1 style="margin:5px 0; color:#0C4A6E;">Profile {cluster_id}</h1>
                    <p style="font-weight:600; color:#0284C7; margin:0;">{profile_dict['auto_description']}</p>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
                **Profile Characteristics**:
                - Represents **{profile_dict['patient_percentage']}%** of ICU patient cohort ({profile_dict['patient_count']} patients)
                - Profile Avg Severity Score: **{profile_dict['avg_severity_score']}**
                - Profile ICU Survival Rate: **{profile_dict['icu_survival_pct']}%**
                
                *This patient is assigned to this profile based on mathematical similarity to the statistical patterns learned from the Kaggle dataset.*
            """)

        st.markdown("### 📊 Patient Values vs Cluster Profile Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 7: ABOUT PROJECT
# =============================================================================
elif selected_page == "7. About Project":
    st.title("ℹ️ About Hospital Patient Profiling Project")
    
    st.markdown("""
    ### 🎯 Project Overview & Objective
    This machine learning application groups Intensive Care Unit (ICU) patients into clinical profiles based on continuous and categorical physiological measurements. Using **Unsupervised Machine Learning (K-Means Clustering)**, the app identifies natural clusters of patient records without outcome bias.

    ### 📊 Kaggle ICU Dataset Specifications
    - **Primary Dataset File**: `train.csv` (4,580 patient records, 16 columns)
    - **Clinical Features used for Clustering (14 features)**:
        - `age`, `gender`, `bmi`, `heart_rate`, `systolic_bp`, `diastolic_bp`, `respiratory_rate`, `oxygen_saturation`, `glucose`, `creatinine`, `wbc_count`, `severity_score`, `ventilation_required`, `chronic_conditions`
    - **Excluded Identifiers & Target**:
        - `patient_id` (Identifier)
        - `survived_icu` (Outcome variable - evaluated only post-clustering to prevent data leakage)

    ### 🛠️ Machine Learning Methodology
    1. **Data Preprocessing**:
        - Median imputation for missing continuous vitals & lab values (`bmi`, `respiratory_rate`, `oxygen_saturation`, `creatinine`, `wbc_count`).
        - Standard scaling (`StandardScaler`) to equalize feature weights during Euclidean distance calculation.
    2. **K-Means Clustering**:
        - Scikit-Learn `KMeans(n_clusters=K, random_state=42, n_init='auto')`.
    3. **Optimal K Selection**:
        - **Elbow Method**: Tracks Inertia (within-cluster sum of squares) reduction.
        - **Silhouette Analysis**: Evaluates cluster separation quality across K=2 to 10.
    4. **Visualization & PCA**:
        - Principal Component Analysis (2D PCA) used **exclusively for graphical projection**.

    ### ☁️ Streamlit Community Cloud Deployment Guide
    1. Push project repository to GitHub.
    2. Ensure `requirements.txt` contains `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `streamlit`, `joblib`.
    3. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
    4. Connect repository and select `app.py` as the entrypoint.
    5. Click **Deploy**.

    ### ⚠️ Medical Disclaimer
    *This application is intended for educational and data-analysis purposes only. Patient profiles are generated using statistical clustering of clinical measurements and should not be interpreted as medical diagnoses, treatment recommendations, or clinical decisions.*
    """)
