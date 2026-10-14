"""
MLflow Experiment Tracking & Model Registry Integration
Demonstrates how PlacementPredict integrates with MLflow (Module M6)
"""
import os
import json
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")

def run_mlflow_experiment():
    print("=" * 60)
    print("MLflow Tracking for PlacementPredict (Module M6)")
    print("=" * 60)

    # 1. Set MLflow Experiment
    mlflow.set_experiment("Placement_Prediction_Benchmark")

    # 2. Load Data
    df = pd.read_csv(DATASET_PATH)
    feature_cols = [
        'Gender', 'CollegeTier', 'Stream', 'Specialisation', 'Hostel',
        'HistoryOfBacklogs', 'CGPA', 'AttendancePercent', 'Internships',
        'Projects', 'Workshops', 'Certifications', 'Publications',
        'AptitudeTestScore', 'SoftSkillsRating', 'CodingTestScore',
        'MockInterviewScore', 'ExtraCurricular'
    ]
    X = df[feature_cols]
    y = df['PlacementStatus']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    # 3. Start MLflow Run
    with mlflow.start_run(run_name="Gradient_Boosting_Champion"):
        # Log Hyperparameters
        params = {
            "n_estimators": 70,
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.8,
            "test_split_size": 0.20,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Train Pipeline
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            subsample=params["subsample"],
            random_state=params["random_state"]
        )
        pipeline = Pipeline([('prep', preprocessor), ('model', clf)])
        pipeline.fit(X_train, y_train)

        # Evaluate
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds))
        auc = float(roc_auc_score(y_test, probs))

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Log Artifacts (Plots)
        roc_plot = os.path.join(BASE_DIR, "static", "generated_plots", "gradient_boosting_roc_curve.png")
        if os.path.exists(roc_plot):
            mlflow.log_artifact(roc_plot, artifact_path="evaluation_plots")

        cm_plot = os.path.join(BASE_DIR, "static", "generated_plots", "gradient_boosting_confusion_matrix.png")
        if os.path.exists(cm_plot):
            mlflow.log_artifact(cm_plot, artifact_path="evaluation_plots")

        # Log & Register the Model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="pickle",
            registered_model_name="PlacementPredictor_GradientBoosting"
        )

        print(f"✓ Run complete! Logged Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        print("To launch the interactive MLflow UI, run: mlflow ui --port 5002")

if __name__ == "__main__":
    run_mlflow_experiment()
