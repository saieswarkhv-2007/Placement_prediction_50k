# ================================================================
# XGBOOST - PLACEMENT PREDICTION
# PyCharm Program
#
# Original / Main Dataset is NOT modified
# All outputs are stored in ONE folder
# ================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Optional XGBoost import with fallback
try:
    from xgboost import XGBClassifier
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier

# ================================================================
# 1. FILE PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "XGBoost_Classifier_Outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================================
# 2. LOAD DATASET
# ================================================================
print("=" * 70)
print("XGBOOST PLACEMENT PREDICTION")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n"
        "Please check the INPUT_FILE path."
    )

original_data = pd.read_csv(INPUT_FILE)
data = original_data.copy()

print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)

# ================================================================
# 3. DISPLAY DATASET INFORMATION
# ================================================================
print("\nDataset columns:", data.columns.tolist())
print("\nFirst 5 records:\n", data.head())
print("\nMissing values:\n", data.isnull().sum())

# ================================================================
# 4. IDENTIFY TARGET COLUMN
# ================================================================
target_candidates = [
    "Placement",
    "Placed",
    "placement",
    "placed",
    "Status",
    "status",
    "Target",
    "target",
    "PlacementStatus",
    "Placement_Status",
    "CGPA_Tier"
]
target_column = None
for column in target_candidates:
    if column in data.columns:
        target_column = column
        break

if target_column is None:
    target_column = data.columns[-1]

print("\nTarget column:", target_column)

# ================================================================
# 5. REMOVE ROWS WITH MISSING TARGET
# ================================================================
data = data.dropna(subset=[target_column]).copy()
X = data.drop(columns=[target_column]).copy()
y = data[target_column].copy()

# ================================================================
# 7. HANDLE CATEGORICAL & NUMERICAL FEATURES
# ================================================================
print("\nProcessing categorical features...")
categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_columns = X.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()

for column in categorical_columns:
    X[column] = X[column].astype(str)
    X[column], _ = pd.factorize(X[column])

# Handle Missing Values in Features
for column in X.columns:
    if X[column].isnull().any():
        if pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].fillna(X[column].median())
        else:
            X[column] = X[column].fillna(X[column].mode()[0])

# ================================================================
# 9. ENCODE TARGET
# ================================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))
class_names = label_encoder.classes_
number_of_classes = len(class_names)

print("\nTarget classes:")
for i, class_name in enumerate(class_names):
    print(f"{i} = {class_name}")

# Save Target Encoding Information
target_encoding = pd.DataFrame({
    "Encoded_Value": range(len(class_names)),
    "Original_Class": class_names
})
target_encoding.to_csv(os.path.join(OUTPUT_FOLDER, "target_encoding.csv"), index=False)

# ================================================================
# 11. TRAIN TEST SPLIT
# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ================================================================
# 12. CREATE XGBOOST MODEL
# ================================================================
print("\nCreating XGBoost model...")
try:
    if number_of_classes == 2:
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    else:
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
except Exception:
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

# ================================================================
# 13. TRAIN XGBOOST
# ================================================================
print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)
print("Training completed successfully.")

# ================================================================
# 14. PREDICTIONS
# ================================================================
y_pred = xgb_model.predict(X_test)
y_probability = xgb_model.predict_proba(X_test) if hasattr(xgb_model, "predict_proba") else None

# ================================================================
# 15. PERFORMANCE METRICS
# ================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 70)
print("XGBOOST PERFORMANCE")
print("=" * 70)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})
metrics.to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_metrics.csv"), index=False)

# Classification Report
classification_report_text = classification_report(y_test, y_pred, target_names=[str(c) for c in class_names], zero_division=0)
print("\nClassification Report:\n", classification_report_text)

with open(os.path.join(OUTPUT_FOLDER, "classification_report.txt"), "w", encoding="utf-8") as file:
    file.write("XGBOOST CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n\n")
    file.write(classification_report_text)

# Actual vs Predicted
actual_labels = label_encoder.inverse_transform(y_test)
predicted_labels = label_encoder.inverse_transform(y_pred)
prediction_results = pd.DataFrame({
    "Actual": actual_labels,
    "Predicted": predicted_labels,
    "Correct": (actual_labels == predicted_labels)
})
if y_probability is not None:
    for i, class_name in enumerate(class_names):
        prediction_results["Probability_" + str(class_name)] = y_probability[:, i]
prediction_results.to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_predictions.csv"), index=False)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_df.to_csv(os.path.join(OUTPUT_FOLDER, "confusion_matrix.csv"))

fig, ax = plt.subplots(figsize=(8, 6))
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
display.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# Feature Importance
if hasattr(xgb_model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    feature_importance.to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_feature_importance.csv"), index=False)

    top_n = min(20, len(feature_importance))
    top_features = feature_importance.head(top_n).sort_values(by="Importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("XGBoost - Top Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "xgboost_feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()

# ROC Curve
try:
    if y_probability is not None:
        plt.figure(figsize=(8, 6))
        if number_of_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_probability[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        else:
            for i in range(number_of_classes):
                binary_y = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_y, y_probability[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("XGBoost - ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "xgboost_roc_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()
except Exception as error:
    print("\nROC curve could not be generated:", error)

# Model Parameters & Dataset Info
parameters = pd.DataFrame({
    "Parameter": ["Algorithm", "Estimators", "Maximum Depth", "Learning Rate", "Subsample", "Column Sample By Tree", "Test Size", "Random State"],
    "Value": ["XGBoost Classifier", 200, 5, 0.05, 0.8, 0.8, 0.20, 42]
})
parameters.to_csv(os.path.join(OUTPUT_FOLDER, "xgboost_parameters.csv"), index=False)

dataset_information = pd.DataFrame({
    "Information": ["Original Rows", "Original Columns", "Rows Used", "Number of Features", "Training Samples", "Testing Samples", "Target Column", "Number of Classes"],
    "Value": [original_data.shape[0], original_data.shape[1], data.shape[0], X.shape[1], X_train.shape[0], X_test.shape[0], target_column, number_of_classes]
})
dataset_information.to_csv(os.path.join(OUTPUT_FOLDER, "dataset_information.csv"), index=False)

# Save Model
try:
    model_file = os.path.join(OUTPUT_FOLDER, "xgboost_placement_model.json")
    xgb_model.save_model(model_file)
except Exception:
    import pickle
    model_file = os.path.join(OUTPUT_FOLDER, "xgboost_placement_model.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(xgb_model, f)

print("\n" + "=" * 70)
print("ALL XGBOOST OUTPUTS SAVED")
print("=" * 70)
print("Output folder:", OUTPUT_FOLDER)
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)
print("ORIGINAL DATASET WAS NOT MODIFIED.")
print("=" * 70)
