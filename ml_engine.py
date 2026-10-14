"""
ML Engine Service for Placement Prediction System
Provides model inference, salary estimation, dataset profiling, and dashboard analytics.
"""
import os
import json
import pickle
import time
from datetime import datetime
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
UPLOAD_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "uploaded_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SUMMARY_FILE = os.path.join(OUTPUTS_DIR, "models_summary.json")
HISTORY_FILE = os.path.join(OUTPUTS_DIR, "prediction_history.json")

# In-memory model cache
_LOADED_MODELS = {}
_SALARY_REGRESSOR = None

FEATURE_COLUMNS = [
    'Gender', 'CollegeTier', 'Stream', 'Specialisation', 'Hostel',
    'HistoryOfBacklogs', 'CGPA', 'AttendancePercent', 'Internships',
    'Projects', 'Workshops', 'Certifications', 'Publications',
    'AptitudeTestScore', 'SoftSkillsRating', 'CodingTestScore',
    'MockInterviewScore', 'ExtraCurricular'
]

def load_models_summary():
    """Loads all model performance summaries from JSON."""
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_model_pipeline(model_id):
    """Loads and caches a scikit-learn / XGBoost pipeline by model_id."""
    global _LOADED_MODELS
    if model_id in _LOADED_MODELS:
        return _LOADED_MODELS[model_id]

    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    if not os.path.exists(model_path):
        # Fallback to gradient_boosting or first available
        for mid in ['gradient_boosting', 'random_forest', 'decision_tree', 'logistic_regression']:
            p = os.path.join(MODELS_DIR, f"{mid}.pkl")
            if os.path.exists(p):
                model_path = p
                model_id = mid
                break

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            pipe = pickle.load(f)
            _LOADED_MODELS[model_id] = pipe
            return pipe
    return None

def get_salary_regressor():
    """Loads the trained salary package regressor."""
    global _SALARY_REGRESSOR
    if _SALARY_REGRESSOR is not None:
        return _SALARY_REGRESSOR

    reg_path = os.path.join(MODELS_DIR, "salary_regressor.pkl")
    if os.path.exists(reg_path):
        with open(reg_path, 'rb') as f:
            _SALARY_REGRESSOR = pickle.load(f)
            return _SALARY_REGRESSOR
    return None

def predict_student(data_dict, model_id='gradient_boosting'):
    """
    Predicts whether a student will get placed.
    Returns: status, probability, salary_package, positive_factors, risk_factors, recommendations.
    """
    pipeline = get_model_pipeline(model_id)
    summary = load_models_summary()
    model_meta = summary.get(model_id, {})
    model_name = model_meta.get('short_name', model_id.replace('_', ' ').title())

    # Build single-row DataFrame with defaults
    cgpa = float(data_dict.get('cgpa', 7.5))
    backlogs = int(data_dict.get('backlogs', 0))
    internships = int(data_dict.get('internships', 1))
    projects = int(data_dict.get('projects', 2))
    workshops = int(data_dict.get('workshops', 1))
    certifications = int(data_dict.get('certifications', 2))
    publications = int(data_dict.get('publications', 0))
    aptitude = float(data_dict.get('aptitude_score', 75))
    coding = float(data_dict.get('coding_score', 80))
    soft_skills = float(data_dict.get('soft_skills', 7.5))
    mock_interview = float(data_dict.get('mock_interview', 75))
    attendance = float(data_dict.get('attendance', 85))
    gender = str(data_dict.get('gender', 'Male'))
    college_tier = str(data_dict.get('college_tier', 'Tier2'))
    stream = str(data_dict.get('stream', 'CS'))
    specialisation = str(data_dict.get('specialisation', 'DataScience'))
    hostel = str(data_dict.get('hostel', 'No'))
    extra_curricular = int(data_dict.get('extra_curricular', 1))

    row_data = {
        'Gender': [gender],
        'CollegeTier': [college_tier],
        'Stream': [stream],
        'Specialisation': [specialisation],
        'Hostel': [hostel],
        'HistoryOfBacklogs': [backlogs],
        'CGPA': [cgpa],
        'AttendancePercent': [attendance],
        'Internships': [internships],
        'Projects': [projects],
        'Workshops': [workshops],
        'Certifications': [certifications],
        'Publications': [publications],
        'AptitudeTestScore': [aptitude],
        'SoftSkillsRating': [soft_skills],
        'CodingTestScore': [coding],
        'MockInterviewScore': [mock_interview],
        'ExtraCurricular': [extra_curricular]
    }
    input_df = pd.DataFrame(row_data)

    # Model inference
    is_placed = 1
    probability = 85.0

    if pipeline is not None:
        try:
            pred = pipeline.predict(input_df)[0]
            is_placed = int(pred)
            if hasattr(pipeline, 'predict_proba'):
                prob_vals = pipeline.predict_proba(input_df)[0]
                probability = float(prob_vals[1] * 100)
            else:
                probability = 90.0 if is_placed == 1 else 20.0
        except Exception as e:
            print("Inference error, using fallback logic:", e)
            # Heuristic calculation based on feature weights
            score = (cgpa / 10.0) * 0.35 + (coding / 100.0) * 0.25 + (aptitude / 100.0) * 0.15 + (internships * 0.1) - (backlogs * 0.15)
            probability = min(99.0, max(5.0, score * 100))
            is_placed = 1 if probability >= 50.0 else 0
    else:
        # Fallback rule
        score = (cgpa / 10.0) * 0.35 + (coding / 100.0) * 0.25 + (aptitude / 100.0) * 0.15 + (internships * 0.1) - (backlogs * 0.15)
        probability = min(99.0, max(5.0, score * 100))
        is_placed = 1 if probability >= 50.0 else 0

    status_str = "Placed" if is_placed == 1 else "Not Placed"
    prob_clean = round(probability, 1)

    # Estimate salary package if placed
    salary_str = "--"
    if is_placed == 1:
        sal_model = get_salary_regressor()
        if sal_model is not None:
            try:
                pred_sal = float(sal_model.predict(input_df)[0])
                pred_sal = max(3.5, round(pred_sal, 2))
                salary_str = f"₹{pred_sal:.1f} LPA"
            except Exception:
                # Heuristic salary based on tier & CGPA
                tier_mult = 1.3 if college_tier == 'Tier1' else (1.0 if college_tier == 'Tier2' else 0.8)
                base = 4.0 + (cgpa - 6.0) * 0.8 + (coding / 100.0) * 2.0
                sal = round(max(3.6, base * tier_mult), 1)
                salary_str = f"₹{sal:.1f} LPA"
        else:
            tier_mult = 1.3 if college_tier == 'Tier1' else (1.0 if college_tier == 'Tier2' else 0.8)
            base = 4.0 + (cgpa - 6.0) * 0.8 + (coding / 100.0) * 2.0
            sal = round(max(3.6, base * tier_mult), 1)
            salary_str = f"₹{sal:.1f} LPA"

    # Analyze Positive & Risk Factors
    positive_factors = []
    risk_factors = []

    if cgpa >= 8.0:
        positive_factors.append(f"Strong Academic Performance: CGPA of {cgpa:.2f} satisfies premier campus hiring criteria.")
    elif cgpa < 6.5:
        risk_factors.append(f"Suboptimal CGPA ({cgpa:.2f}): Below standard shortlisting threshold for tier-1 companies.")

    if coding >= 80:
        positive_factors.append(f"High Coding Proficiency: Score of {coding:.0f}/100 excels technical screening.")
    elif coding < 55:
        risk_factors.append(f"Low Technical Benchmark: Coding score ({coding:.0f}/100) requires intensive algorithm practice.")

    if internships >= 2:
        positive_factors.append(f"Demonstrated Industry Readiness: {internships} verified internships.")
    elif internships == 0:
        risk_factors.append("No Internship Experience: Limits practical domain credibility.")

    if backlogs > 0:
        risk_factors.append(f"History of Backlogs: {backlogs} backlog(s) may trigger corporate policy eligibility filters.")
    else:
        positive_factors.append("Clean Academic Record: 0 backlogs eliminates automated eligibility rejections.")

    if aptitude >= 80:
        positive_factors.append(f"Strong Quantitative & Reasoning Skill: Aptitude score {aptitude:.0f}/100.")

    if projects >= 3:
        positive_factors.append(f"Substantial Project Portfolio: {projects} completed projects.")

    if not positive_factors:
        positive_factors.append("Balanced baseline qualifications across core academic categories.")
    if not risk_factors:
        risk_factors.append("No significant bottlenecks identified in current profile.")

    # Tailored recommendations
    recommendations = []
    if is_placed == 1:
        recommendations.append("Continue practicing advanced LeetCode / competitive programming to target premium product company offers.")
        recommendations.append("Refine technical system design communication for executive mock rounds.")
        recommendations.append("Highlight high-impact metrics in project portfolio (latency reduction, deployment pipelines).")
    else:
        if backlogs > 0:
            recommendations.append("Clear all pending backlogs as immediate priority before campus recruitment drives commence.")
        if coding < 75:
            recommendations.append("Dedicate 2 hours daily to Data Structures & Algorithms (Arrays, Dynamic Programming, Trees).")
        if internships == 0:
            recommendations.append("Pursue a 2-3 month virtual or industry internship or open-source software contribution.")
        if projects < 2:
            recommendations.append("Build 2 full-stack or end-to-end ML applications with live deployed URLs on GitHub.")

    # Record history
    history_record = {
        "id": f"PRED-{int(time.time()*1000)%1000000:06d}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_id": model_id,
        "model_name": model_name,
        "cgpa": cgpa,
        "stream": stream,
        "college_tier": college_tier,
        "backlogs": backlogs,
        "internships": internships,
        "coding_score": coding,
        "probability": prob_clean,
        "status": status_str,
        "salary_package": salary_str
    }
    save_prediction_record(history_record)

    return {
        "status": status_str,
        "is_placed": is_placed,
        "probability": prob_clean,
        "salary_package": salary_str,
        "model_id": model_id,
        "model_name": model_name,
        "positive_factors": positive_factors,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "record": history_record
    }

def get_prediction_history():
    """Retrieves list of past predictions."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_prediction_record(record):
    """Saves a new prediction to the persistent history."""
    history = get_prediction_history()
    history.insert(0, record)
    # Keep top 100 records
    history = history[:100]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def clear_prediction_history():
    """Clears past predictions history."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f, indent=2)

def get_dashboard_analytics():
    """Computes summary KPIs and distribution data for dashboard."""
    history = get_prediction_history()

    if not history:
        return {
            'total_predictions': 0,
            'placed_count': 0,
            'not_placed_count': 0,
            'placement_rate': 0.0,
            'avg_probability': 0.0,
            'avg_cgpa': 0.0,
            'top_model': 'Gradient Boosting',
            'model_usage': {},
            'scatter_data': [],
            'recent_history': []
        }

    total = len(history)
    placed = sum(1 for h in history if h.get('status') == 'Placed')
    not_placed = total - placed
    rate = round((placed / total) * 100, 1) if total > 0 else 0.0
    avg_prob = round(sum(h.get('probability', 0) for h in history) / total, 1)
    avg_cgpa = round(sum(h.get('cgpa', 0) for h in history) / total, 2)

    # Model usage counts
    usage = {}
    for h in history:
        m = h.get('model_name', 'Unknown')
        usage[m] = usage.get(m, 0) + 1

    top_model = max(usage.items(), key=lambda x: x[1])[0] if usage else 'Gradient Boosting'

    # Scatter points (CGPA, Probability, Status)
    scatter = [
        {
            'cgpa': h.get('cgpa', 7.0),
            'prob': h.get('probability', 50.0),
            'status': h.get('status', 'Placed'),
            'model': h.get('model_name', '')
        }
        for h in history
    ]

    return {
        'total_predictions': total,
        'placed_count': placed,
        'not_placed_count': not_placed,
        'placement_rate': rate,
        'avg_probability': avg_prob,
        'avg_cgpa': avg_cgpa,
        'top_model': top_model,
        'model_usage': usage,
        'scatter_data': scatter,
        'recent_history': history[:15]
    }

def get_dataset_profile(filepath=None, preview_rows=15):
    """Inspects dataset and generates column-by-column statistics and preview."""
    target_path = filepath if filepath and os.path.exists(filepath) else (
        UPLOAD_DATASET_PATH if os.path.exists(UPLOAD_DATASET_PATH) else DATASET_PATH
    )

    if not os.path.exists(target_path):
        return None

    try:
        df = pd.read_csv(target_path)
    except Exception as e:
        print("Error reading dataset:", e)
        return None

    total_rows = len(df)
    total_cols = len(df.columns)
    missing_total = int(df.isnull().sum().sum())
    duplicate_total = int(df.duplicated().sum())

    # Target column analysis if present
    target_dist = {}
    if 'PlacementStatus' in df.columns:
        counts = df['PlacementStatus'].value_counts().to_dict()
        placed = int(counts.get(1, counts.get('1', 0)))
        unplaced = int(counts.get(0, counts.get('0', 0)))
        pct = round((placed / (placed + unplaced)) * 100, 1) if (placed + unplaced) > 0 else 0
        target_dist = {
            'placed': placed,
            'not_placed': unplaced,
            'placement_percentage': pct
        }

    # Column summary
    col_summaries = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_rows) * 100, 1)
        unique_count = int(df[col].nunique())

        stat_info = {}
        if pd.api.types.is_numeric_dtype(df[col]):
            stat_info = {
                'min': round(float(df[col].min()), 2) if not pd.isna(df[col].min()) else '-',
                'max': round(float(df[col].max()), 2) if not pd.isna(df[col].max()) else '-',
                'mean': round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else '-'
            }
        else:
            top_vals = df[col].dropna().value_counts().index[:3].tolist()
            stat_info = {'top_values': ", ".join([str(v) for v in top_vals])}

        col_summaries.append({
            'name': col,
            'dtype': dtype,
            'null_count': null_count,
            'null_pct': null_pct,
            'unique_count': unique_count,
            'stats': stat_info
        })

    # Preview rows
    preview = df.head(preview_rows).fillna("-").to_dict(orient='records')
    columns = df.columns.tolist()

    return {
        'filename': os.path.basename(target_path),
        'is_custom_upload': (target_path == UPLOAD_DATASET_PATH),
        'total_rows': total_rows,
        'total_cols': total_cols,
        'missing_total': missing_total,
        'duplicate_total': duplicate_total,
        'target_dist': target_dist,
        'columns': columns,
        'column_summaries': col_summaries,
        'preview_data': preview
    }
