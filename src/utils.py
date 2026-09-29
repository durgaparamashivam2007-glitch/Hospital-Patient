import os
import pandas as pd

DEFAULT_CLINICAL_FEATURES = [
    'age',
    'gender',
    'bmi',
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'respiratory_rate',
    'oxygen_saturation',
    'glucose',
    'creatinine',
    'wbc_count',
    'severity_score',
    'ventilation_required',
    'chronic_conditions'
]

ALL_REQUIRED_COLUMNS = [
    'patient_id',
    'age',
    'gender',
    'bmi',
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'respiratory_rate',
    'oxygen_saturation',
    'glucose',
    'creatinine',
    'wbc_count',
    'severity_score',
    'ventilation_required',
    'chronic_conditions',
    'survived_icu'
]

EXCLUDED_FEATURES = ['patient_id', 'survived_icu']

FEATURE_LABELS = {
    'age': 'Age (years)',
    'gender': 'Gender (0=Female, 1=Male)',
    'bmi': 'BMI (kg/m²)',
    'heart_rate': 'Heart Rate (bpm)',
    'systolic_bp': 'Systolic BP (mmHg)',
    'diastolic_bp': 'Diastolic BP (mmHg)',
    'respiratory_rate': 'Respiratory Rate (bpm)',
    'oxygen_saturation': 'Oxygen Saturation (%)',
    'glucose': 'Glucose (mg/dL)',
    'creatinine': 'Creatinine (mg/dL)',
    'wbc_count': 'WBC Count (K/µL)',
    'severity_score': 'Severity Score',
    'ventilation_required': 'Ventilation Required (0/1)',
    'chronic_conditions': 'Chronic Conditions Count',
    'survived_icu': 'ICU Survival (0/1)'
}

def find_data_path():
    """Search for train.csv in standard relative path locations."""
    candidate_paths = [
        os.path.join('data', 'train.csv'),
        'train.csv',
        os.path.join('..', 'data', 'train.csv'),
        os.path.join('..', 'train.csv')
    ]
    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    return candidate_paths[0]

def load_dataset(data_path=None):
    """
    Load train.csv dataset safely.
    Returns (df, None) on success or (None, error_message) on failure.
    """
    if data_path is None:
        data_path = find_data_path()
        
    if not os.path.exists(data_path):
        return None, f"Dataset file not found at path: {data_path}. Please place 'train.csv' inside the 'data/' directory."
    
    try:
        df = pd.read_csv(data_path)
        is_valid, msg = validate_dataset(df)
        if not is_valid:
            return None, msg
        return df, None
    except Exception as e:
        return None, f"Failed to read dataset file: {str(e)}"

def validate_dataset(df):
    """
    Validate the loaded patient dataset against required specifications.
    """
    if df is None or df.empty:
        return False, "The dataset is empty or could not be loaded."
    
    missing_cols = [col for col in ALL_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns in dataset: {', '.join(missing_cols)}"
    
    if len(df) < 10:
        return False, f"Insufficient rows in dataset for clustering (found {len(df)} rows, expected >= 10)."
    
    return True, "Dataset validation successful."

def ensure_directories(base_path='.'):
    """Ensure data/ and models/ directories exist."""
    models_dir = os.path.join(base_path, 'models')
    data_dir = os.path.join(base_path, 'data')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return models_dir, data_dir
