import os
import joblib
import numpy as np
import pandas as pd

def compute_cluster_profiles(df_imputed, labels, feature_cols):
    """
    Compute cluster statistics on imputed original-scale patient data.
    
    Includes clinical feature averages, ventilation %, and ICU survival % (outcome variable).
    """
    df_temp = df_imputed.copy()
    df_temp['cluster'] = labels
    
    total_patients = len(df_temp)
    cluster_ids = sorted(df_temp['cluster'].unique())
    
    profile_rows = []
    
    # Calculate overall population mean and std for feature comparison
    pop_stats = {}
    for col in feature_cols:
        pop_stats[col] = {
            'mean': df_temp[col].mean(),
            'std': df_temp[col].std() if df_temp[col].std() > 0 else 1.0
        }
        
    for c_id in cluster_ids:
        c_df = df_temp[df_temp['cluster'] == c_id]
        c_count = len(c_df)
        c_pct = (c_count / total_patients) * 100.0
        
        row = {
            'cluster_id': c_id,
            'profile_name': f"Profile {c_id}",
            'patient_count': c_count,
            'patient_percentage': round(c_pct, 1),
            'avg_age': round(c_df['age'].mean(), 1),
            'avg_bmi': round(c_df['bmi'].mean(), 1),
            'avg_heart_rate': round(c_df['heart_rate'].mean(), 1),
            'avg_systolic_bp': round(c_df['systolic_bp'].mean(), 1),
            'avg_diastolic_bp': round(c_df['diastolic_bp'].mean(), 1),
            'avg_respiratory_rate': round(c_df['respiratory_rate'].mean(), 1),
            'avg_oxygen_saturation': round(c_df['oxygen_saturation'].mean(), 1),
            'avg_glucose': round(c_df['glucose'].mean(), 1),
            'avg_creatinine': round(c_df['creatinine'].mean(), 2),
            'avg_wbc_count': round(c_df['wbc_count'].mean(), 1),
            'avg_severity_score': round(c_df['severity_score'].mean(), 1),
            'ventilation_pct': round((c_df['ventilation_required'].mean()) * 100.0, 1),
            'avg_chronic_conditions': round(c_df['chronic_conditions'].mean(), 1)
        }
        
        # Include survived_icu outcome analysis if present in df
        if 'survived_icu' in c_df.columns:
            row['icu_survival_pct'] = round((c_df['survived_icu'].mean()) * 100.0, 1)
        else:
            row['icu_survival_pct'] = np.nan
            
        # Store individual feature averages for Z-score calculation
        for col in feature_cols:
            row[f'feat_mean_{col}'] = c_df[col].mean()
            
        profile_rows.append(row)
        
    profiles_df = pd.DataFrame(profile_rows)
    
    # Generate automatic descriptions
    descriptions = generate_auto_descriptions(profiles_df, pop_stats, feature_cols)
    profiles_df['auto_description'] = descriptions
    
    return profiles_df, pop_stats

def generate_auto_descriptions(profiles_df, pop_stats, feature_cols):
    """
    Generate evidence-based profile descriptions based on statistical z-scores.
    """
    descriptions = []
    
    for idx, row in profiles_df.iterrows():
        z_scores = {}
        for col in feature_cols:
            pop_mean = pop_stats[col]['mean']
            pop_std = pop_stats[col]['std']
            c_mean = row[f'feat_mean_{col}']
            z = (c_mean - pop_mean) / pop_std
            z_scores[col] = z
            
        # Identify top elevated and reduced clinical indicators
        sorted_z = sorted(z_scores.items(), key=lambda x: x[1], reverse=True)
        top_high = [item for item in sorted_z if item[1] >= 0.35]
        top_low = [item for item in sorted_z if item[1] <= -0.35]
        
        desc_parts = []
        
        # Check specific vital groups
        if any(item[0] == 'glucose' for item in top_high):
            desc_parts.append("Higher glucose")
        if any(item[0] in ['systolic_bp', 'diastolic_bp'] for item in top_high):
            desc_parts.append("Higher blood pressure")
        if any(item[0] == 'oxygen_saturation' for item in top_low):
            desc_parts.append("Lower oxygen saturation")
        if any(item[0] == 'severity_score' for item in top_high):
            desc_parts.append("Higher severity score")
        if any(item[0] == 'age' for item in top_high):
            desc_parts.append("Older age demographic")
        if any(item[0] == 'bmi' for item in top_high):
            desc_parts.append("Elevated BMI")
        if any(item[0] in ['creatinine', 'wbc_count'] for item in top_high):
            desc_parts.append("Elevated lab markers")
            
        if desc_parts:
            # Join up to 2 key characteristics
            desc = " & ".join(desc_parts[:2]) + " profile"
        else:
            desc = "Mixed clinical measurement profile"
            
        descriptions.append(desc)
        
    return descriptions

def predict_patient_profile(patient_dict, pipeline, model, profiles_df, feature_cols, pop_stats):
    """
    Assign a new patient to a cluster profile and compare measurements.
    
    Returns:
    - assigned_cluster_id: int
    - profile_info: dict of assigned cluster statistics
    - comparison_df: DataFrame comparing Patient Value, Cluster Avg, Difference, Population Avg
    """
    # Create single-row dataframe in correct feature order
    input_df = pd.DataFrame([patient_dict])[feature_cols]
    
    # Transform with pipeline
    X_new_scaled = pipeline.transform(input_df)
    
    # Predict cluster
    cluster_id = int(model.predict(X_new_scaled)[0])
    
    # Get cluster stats row
    c_profile = profiles_df[profiles_df['cluster_id'] == cluster_id].iloc[0]
    
    # Build feature comparison matrix
    comp_rows = []
    feature_labels = {
        'age': 'Age (years)',
        'gender': 'Gender (0=F, 1=M)',
        'bmi': 'BMI (kg/m²)',
        'heart_rate': 'Heart Rate (bpm)',
        'systolic_bp': 'Systolic BP (mmHg)',
        'diastolic_bp': 'Diastolic BP (mmHg)',
        'respiratory_rate': 'Resp Rate (bpm)',
        'oxygen_saturation': 'Oxygen Sat (%)',
        'glucose': 'Glucose (mg/dL)',
        'creatinine': 'Creatinine (mg/dL)',
        'wbc_count': 'WBC Count (K/µL)',
        'severity_score': 'Severity Score',
        'ventilation_required': 'Ventilation (0/1)',
        'chronic_conditions': 'Chronic Conditions'
    }
    
    for col in feature_cols:
        val = float(patient_dict[col])
        c_avg = float(c_profile[f'feat_mean_{col}'])
        diff = val - c_avg
        pop_avg = float(pop_stats[col]['mean'])
        
        comp_rows.append({
            'Feature': feature_labels.get(col, col),
            'Patient Value': round(val, 2),
            'Cluster Average': round(c_avg, 2),
            'Difference (Value - Avg)': round(diff, 2),
            'Overall Population Avg': round(pop_avg, 2)
        })
        
    comparison_df = pd.DataFrame(comp_rows)
    return cluster_id, c_profile.to_dict(), comparison_df

def save_model_artifacts(model, pipeline, profiles_df, feature_cols, models_dir='models'):
    """Save model and pipeline artifacts using joblib."""
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_dir, 'clustering_model.joblib'))
    joblib.dump(pipeline, os.path.join(models_dir, 'preprocessing_pipeline.joblib'))
    joblib.dump(profiles_df, os.path.join(models_dir, 'cluster_profiles.joblib'))
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_config.joblib'))

def load_model_artifacts(models_dir='models'):
    """Load joblib models and return (model, pipeline, profiles_df, feature_cols) or None."""
    model_path = os.path.join(models_dir, 'clustering_model.joblib')
    pipe_path = os.path.join(models_dir, 'preprocessing_pipeline.joblib')
    prof_path = os.path.join(models_dir, 'cluster_profiles.joblib')
    feat_path = os.path.join(models_dir, 'feature_config.joblib')
    
    if os.path.exists(model_path) and os.path.exists(pipe_path) and os.path.exists(prof_path) and os.path.exists(feat_path):
        try:
            model = joblib.load(model_path)
            pipeline = joblib.load(pipe_path)
            profiles_df = joblib.load(prof_path)
            feature_cols = joblib.load(feat_path)
            return model, pipeline, profiles_df, feature_cols
        except Exception:
            return None
    return None
