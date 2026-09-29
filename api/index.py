import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template_string

# Add project root directory to sys.path to allow src package imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.utils import (
    load_dataset,
    DEFAULT_CLINICAL_FEATURES,
    FEATURE_LABELS,
    ensure_directories
)
from src.data_preprocessing import preprocess_dataset
from src.clustering import train_kmeans
from src.profiling import compute_cluster_profiles, predict_patient_profile

ensure_directories(base_path=ROOT_DIR)

app = Flask(__name__)
application = app
handler = app

# Cache data loading & model fitting for serverless execution
def get_model_and_profiles(k=4):
    df, err = load_dataset()
    if err or df is None:
        return None, None, None, None, None, err or "Dataset unavailable"
    
    feature_cols = DEFAULT_CLINICAL_FEATURES
    X_scaled, pipeline, df_imputed, feature_names = preprocess_dataset(df, feature_cols)
    kmeans_model, labels, inertia, sil_score = train_kmeans(X_scaled, n_clusters=k)
    profiles_df, pop_stats = compute_cluster_profiles(df_imputed, labels, feature_cols)
    
    return df, pipeline, kmeans_model, profiles_df, pop_stats, None

HTML_LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hospital Patient Profiling | Vercel Deployment</title>
    <style>
        :root {
            --primary: #0284C7;
            --primary-dark: #0369A1;
            --bg-color: #F8FAFC;
            --card-bg: #FFFFFF;
            --text-dark: #0F172A;
            --text-muted: #64748B;
            --border-color: #E2E8F0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-dark);
            line-height: 1.6;
            padding: 24px;
        }
        .container {
            max-width: 1100px;
            margin: 0 auto;
        }
        header {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            color: white;
            padding: 36px 40px;
            border-radius: 16px;
            margin-bottom: 32px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        }
        header h1 { font-size: 2.2rem; font-weight: 800; margin-bottom: 8px; }
        header p { color: #94A3B8; font-size: 1.1rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 32px; }
        .card {
            background: var(--card-bg);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        .card h3 { font-size: 1.2rem; margin-bottom: 12px; color: var(--primary-dark); }
        .badge {
            display: inline-block;
            background: #E0F2FE;
            color: var(--primary-dark);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        pre {
            background: #1E293B;
            color: #F8FAFC;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9rem;
        }
        .endpoint-table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        .endpoint-table th, .endpoint-table td {
            text-align: left; padding: 12px; border-bottom: 1px solid var(--border-color);
        }
        .endpoint-table th { background: #F1F5F9; color: var(--text-dark); }
        .method-get { background: #DEF7EC; color: #03543F; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.8rem; }
        .method-post { background: #E1EFFE; color: #1E40AF; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.8rem; }
        footer { margin-top: 40px; text-align: center; color: var(--text-muted); font-size: 0.9rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏥 Hospital Patient Profiling API</h1>
            <p>Unsupervised Machine Learning & Patient Segmentation System | Deployed on Vercel Serverless</p>
        </header>

        <div class="grid">
            <div class="card">
                <h3>📊 System Status</h3>
                <p>Status: <span class="badge">Online & Ready</span></p>
                <p>Dataset Cohort: <strong>4,580 ICU Patients</strong></p>
                <p>Clustering Model: <strong>K-Means (K=4)</strong></p>
            </div>
            <div class="card">
                <h3>⚡ Serverless REST API</h3>
                <p>Full REST Endpoints for programmatic patient profile prediction, dataset metrics, and cluster summaries.</p>
            </div>
            <div class="card">
                <h3>🚀 Interactive Web UI</h3>
                <p>For the full multi-page interactive Streamlit dashboard with Plotly charts and sliders, run locally or host on Streamlit Cloud.</p>
            </div>
        </div>

        <div class="card">
            <h3>🔗 Available API Endpoints</h3>
            <table class="endpoint-table">
                <thead>
                    <tr>
                        <th>Method</th>
                        <th>Endpoint</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="method-get">GET</span></td>
                        <td><code>/api/health</code></td>
                        <td>Check API health and model readiness</td>
                    </tr>
                    <tr>
                        <td><span class="method-get">GET</span></td>
                        <td><code>/api/profiles</code></td>
                        <td>Retrieve learned patient cluster profile summaries</td>
                    </tr>
                    <tr>
                        <td><span class="method-get">GET</span></td>
                        <td><code>/api/dataset</code></td>
                        <td>Get dataset statistics and column metadata</td>
                    </tr>
                    <tr>
                        <td><span class="method-post">POST</span></td>
                        <td><code>/api/predict</code></td>
                        <td>Classify a new patient's vitals into a patient profile</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <br>
        <div class="card">
            <h3>💡 Example POST Request to <code>/api/predict</code></h3>
            <pre>curl -X POST https://your-app.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60, "gender": 1, "bmi": 28.5, "heart_rate": 85,
    "systolic_bp": 125, "diastolic_bp": 80, "respiratory_rate": 20,
    "oxygen_saturation": 96, "glucose": 140, "creatinine": 1.2,
    "wbc_count": 11.0, "severity_score": 12, "ventilation_required": 0,
    "chronic_conditions": 1
  }'</pre>
        </div>

        <footer>
            Hospital Patient Profiling &copy; 2026 | Built with Python, Scikit-Learn, Flask & Streamlit
        </footer>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_LANDING_PAGE)

@app.route('/api/health', methods=['GET'])
def health():
    df, pipeline, kmeans_model, profiles_df, pop_stats, err = get_model_and_profiles()
    if err:
        return jsonify({"status": "error", "message": err}), 500
    return jsonify({
        "status": "healthy",
        "service": "Hospital Patient Profiling API",
        "patient_records": len(df),
        "features": len(DEFAULT_CLINICAL_FEATURES),
        "clusters": len(profiles_df)
    })

@app.route('/api/profiles', methods=['GET'])
def profiles():
    df, pipeline, kmeans_model, profiles_df, pop_stats, err = get_model_and_profiles()
    if err:
        return jsonify({"error": err}), 500
    
    clean_profiles = profiles_df[['cluster_id', 'profile_name', 'auto_description', 'patient_count', 'patient_percentage', 'avg_age', 'avg_glucose', 'avg_systolic_bp', 'avg_oxygen_saturation', 'avg_severity_score', 'icu_survival_pct']].to_dict(orient='records')
    return jsonify({
        "clusters": clean_profiles
    })

@app.route('/api/dataset', methods=['GET'])
def dataset_stats():
    df, _, _, _, _, err = get_model_and_profiles()
    if err:
        return jsonify({"error": err}), 500
    
    stats = {
        "total_records": len(df),
        "columns": list(df.columns),
        "clinical_features": DEFAULT_CLINICAL_FEATURES,
        "missing_counts": df[DEFAULT_CLINICAL_FEATURES].isnull().sum().to_dict()
    }
    return jsonify(stats)

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body provided"}), 400
        
    df, pipeline, kmeans_model, profiles_df, pop_stats, err = get_model_and_profiles()
    if err:
        return jsonify({"error": err}), 500
        
    try:
        cluster_id, profile_dict, comp_df = predict_patient_profile(
            data, pipeline, kmeans_model, profiles_df, DEFAULT_CLINICAL_FEATURES, pop_stats
        )
        return jsonify({
            "assigned_cluster": int(cluster_id),
            "profile_name": profile_dict['profile_name'],
            "auto_description": profile_dict['auto_description'],
            "population_percentage": profile_dict['patient_percentage'],
            "avg_severity_score": profile_dict['avg_severity_score'],
            "icu_survival_pct": profile_dict['icu_survival_pct'],
            "feature_comparison": comp_df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
