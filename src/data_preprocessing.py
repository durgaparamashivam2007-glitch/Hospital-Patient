import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(num_cols, cat_cols=None):
    """
    Build a scikit-learn ColumnTransformer preprocessing pipeline.
    
    Numerical features: Median Imputation -> StandardScaler
    Categorical features: Most Frequent Imputation -> OneHotEncoder
    """
    if cat_cols is None:
        cat_cols = []
        
    transformers = []
    
    if num_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, num_cols))
        
    if cat_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, cat_cols))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor

def preprocess_dataset(df, feature_cols, num_cols=None, cat_cols=None):
    """
    Process dataset with imputation and scaling.
    
    Returns:
    - X_scaled: numpy array of scaled features
    - pipeline: fitted ColumnTransformer pipeline
    - df_imputed: pandas DataFrame with missing values imputed (original scale)
    - feature_names: feature names after transformation
    """
    if num_cols is None and cat_cols is None:
        # Separate columns by data type in feature_cols
        num_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
        cat_cols = [c for c in feature_cols if c not in num_cols]
        
    pipeline = build_preprocessing_pipeline(num_cols, cat_cols)
    X_scaled = pipeline.fit_transform(df[feature_cols])
    
    # Generate imputed raw dataframe for interpretable cluster stats
    df_imputed = df.copy()
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        df_imputed[num_cols] = num_imputer.fit_transform(df[num_cols])
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        
    # Get transformed feature names
    feature_names = list(num_cols)
    if cat_cols and hasattr(pipeline.named_transformers_['cat'], 'named_steps'):
        encoder = pipeline.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_names = encoder.get_feature_names_out(cat_cols)
        feature_names.extend(encoded_cat_names)
        
    return X_scaled, pipeline, df_imputed, feature_names

def inspect_preprocessing_summary(df, feature_cols):
    """
    Generate summary diagnostics for missing values, shape, data types before/after.
    """
    total_rows = len(df)
    missing_before = df[feature_cols].isnull().sum()
    duplicate_rows = df.duplicated().sum()
    
    summary = {
        'total_rows': total_rows,
        'duplicate_rows': duplicate_rows,
        'missing_before': missing_before,
        'missing_total': missing_before.sum(),
        'feature_count': len(feature_cols)
    }
    return summary
