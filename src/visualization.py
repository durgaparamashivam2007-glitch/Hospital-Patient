import plotly.express as px
import plotly.graph_objects as gg
import pandas as pd
import numpy as np

# Professional Neutral Color Palette (Slate, Teal, Amber, Coral, Purple, Mint, Warm Gray)
CLUSTER_COLORS = [
    '#2E7D32', # Deep Emerald
    '#00838F', # Muted Cyan
    '#D81B60', # Mulberry/Rose
    '#E65100', # Warm Amber
    '#6A1B9A', # Deep Violet
    '#455A64', # Slate Gray
    '#C0CA33', # Olive Green
    '#8D6E63', # Warm Taupe
    '#00796B', # Deep Teal
    '#5C6BC0'  # Slate Blue
]

CUSTOM_TEMPLATE = "plotly_white"

def plot_elbow_curve(k_df, optimal_k=None):
    """Generate Elbow Method (Inertia vs K) interactive line plot."""
    fig = px.line(
        k_df,
        x='k',
        y='inertia',
        markers=True,
        title="<b>Elbow Method for Optimal K</b> (Within-Cluster Sum of Squares)",
        labels={'k': 'Number of Clusters (K)', 'inertia': 'Inertia (Sum of Squared Distances)'},
        template=CUSTOM_TEMPLATE
    )
    fig.update_traces(line_color='#37474F', marker=dict(size=9, color='#00838F'))
    
    if optimal_k is not None:
        opt_row = k_df[k_df['k'] == optimal_k].iloc[0]
        fig.add_scatter(
            x=[optimal_k],
            y=[opt_row['inertia']],
            mode='markers+text',
            marker=dict(size=14, color='#2E7D32', symbol='star'),
            text=[f" Optimal K={optimal_k}"],
            textposition="top right",
            name="Suggested Optimal K"
        )
        
    fig.update_layout(
        font=dict(family="sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_silhouette_scores(k_df, optimal_k=None):
    """Generate Silhouette Score vs K interactive line plot."""
    fig = px.line(
        k_df,
        x='k',
        y='silhouette_score',
        markers=True,
        title="<b>Silhouette Score Analysis</b> (Higher is Better)",
        labels={'k': 'Number of Clusters (K)', 'silhouette_score': 'Silhouette Score'},
        template=CUSTOM_TEMPLATE
    )
    fig.update_traces(line_color='#455A64', marker=dict(size=9, color='#2E7D32'))
    
    if optimal_k is not None:
        opt_row = k_df[k_df['k'] == optimal_k].iloc[0]
        fig.add_scatter(
            x=[optimal_k],
            y=[opt_row['silhouette_score']],
            mode='markers+text',
            marker=dict(size=14, color='#2E7D32', symbol='star'),
            text=[f" Max Score ({opt_row['silhouette_score']:.3f})"],
            textposition="top right",
            name="Suggested K"
        )
        
    fig.update_layout(
        font=dict(family="sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_cluster_distribution(profiles_df):
    """Generate Patient Count & Percentage per Profile bar chart."""
    fig = px.bar(
        profiles_df,
        x='profile_name',
        y='patient_count',
        color='profile_name',
        color_discrete_sequence=CLUSTER_COLORS,
        text='patient_percentage',
        title="<b>Patient Profile Distribution</b>",
        labels={'profile_name': 'Patient Profile', 'patient_count': 'Number of Patients'},
        template=CUSTOM_TEMPLATE
    )
    fig.update_traces(
        texttemplate='%{text}% of total',
        textposition='outside'
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_pca_2d(pca_df, labels, df_raw):
    """
    Generate 2D PCA Scatter Plot showing Patient Clusters.
    Hover tooltip shows: patient_id, age, bmi, glucose, systolic_bp, oxygen_saturation, severity_score, cluster.
    """
    plot_df = pca_df.copy()
    plot_df['Cluster'] = [f"Profile {l}" for l in labels]
    plot_df['patient_id'] = df_raw['patient_id'].values
    plot_df['age'] = df_raw['age'].values
    plot_df['bmi'] = df_raw['bmi'].values
    plot_df['glucose'] = df_raw['glucose'].values
    plot_df['systolic_bp'] = df_raw['systolic_bp'].values
    plot_df['oxygen_saturation'] = df_raw['oxygen_saturation'].values
    plot_df['severity_score'] = df_raw['severity_score'].values
    
    var_explained = plot_df['explained_variance_ratio'].iloc[0] * 100
    
    fig = px.scatter(
        plot_df,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        color_discrete_sequence=CLUSTER_COLORS,
        hover_data={
            'patient_id': True,
            'age': ':.0f',
            'bmi': ':.1f',
            'glucose': ':.0f',
            'systolic_bp': ':.0f',
            'oxygen_saturation': ':.1f',
            'severity_score': ':.0f',
            'PCA1': False,
            'PCA2': False
        },
        title=f"<b>2D PCA Cluster Projection</b> (Explains {var_explained:.1f}% Variance)",
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        template=CUSTOM_TEMPLATE
    )
    
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(
        legend_title="Patient Profile",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_profile_heatmap(profiles_df, feature_cols):
    """
    Generate heatmap of standardized mean feature values across clusters.
    """
    heatmap_data = []
    
    # Calculate global feature means and stds across all profile means
    for col in feature_cols:
        means = [profiles_df.loc[i, f'feat_mean_{col}'] for i in range(len(profiles_df))]
        std_val = np.std(means) if np.std(means) > 0 else 1.0
        z_vals = [(m - np.mean(means)) / std_val for m in means]
        heatmap_data.append(z_vals)
        
    feature_labels = [col.replace('_', ' ').title() for col in feature_cols]
    cluster_names = profiles_df['profile_name'].tolist()
    
    fig = px.imshow(
        heatmap_data,
        x=cluster_names,
        y=feature_labels,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="<b>Relative Clinical Feature Intensity Heatmap</b> (Normalized Z-Scores)",
        labels=dict(x="Patient Profile", y="Clinical Feature", color="Relative Intensity"),
        template=CUSTOM_TEMPLATE
    )
    
    fig.update_layout(
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_feature_comparisons(profiles_df, feature_cols):
    """Generate grouped comparison bar charts for vital signs."""
    vitals = ['avg_heart_rate', 'avg_systolic_bp', 'avg_diastolic_bp', 'avg_oxygen_saturation', 'avg_glucose', 'avg_severity_score']
    vitals_present = [v for v in vitals if v in profiles_df.columns]
    
    fig = gg.Figure()
    
    for i, row in profiles_df.iterrows():
        fig.add_trace(gg.Bar(
            name=row['profile_name'],
            x=['Heart Rate', 'Systolic BP', 'Diastolic BP', 'O2 Sat (%)', 'Glucose', 'Severity Score'],
            y=[row['avg_heart_rate'], row['avg_systolic_bp'], row['avg_diastolic_bp'], row['avg_oxygen_saturation'], row['avg_glucose'], row['avg_severity_score']],
            marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        ))
        
    fig.update_layout(
        barmode='group',
        title="<b>Key Clinical Measurements Across Profiles</b>",
        xaxis_title="Clinical Measurement",
        yaxis_title="Average Value",
        template=CUSTOM_TEMPLATE,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_survival_by_cluster(profiles_df):
    """Generate ICU Survival % by Cluster bar chart (Post-clustering descriptive analysis)."""
    if 'icu_survival_pct' not in profiles_df.columns:
        return None
        
    fig = px.bar(
        profiles_df,
        x='profile_name',
        y='icu_survival_pct',
        color='profile_name',
        color_discrete_sequence=CLUSTER_COLORS,
        text='icu_survival_pct',
        title="<b>ICU Survival Rate by Patient Profile</b> (Post-Clustering Analysis)",
        labels={'profile_name': 'Patient Profile', 'icu_survival_pct': 'ICU Survival Rate (%)'},
        template=CUSTOM_TEMPLATE
    )
    
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    fig.update_layout(
        yaxis=dict(range=[0, 110]),
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
