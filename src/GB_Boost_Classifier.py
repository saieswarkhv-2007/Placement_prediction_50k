# ================================================================
# GRADIENT BOOSTING MODEL - PLACEMENT PREDICTION
# PyCharm Program
# Original dataset is NOT modified
# All outputs are stored in ONE folder
# ================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
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

# ================================================================
# 1. PATH SETTINGS
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "GBoost_Classifier_Outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 70)
print("GRADIENT BOOSTING - PLACEMENT PREDICTION")
print("=" * 70)

# ================================================================
# 2. LOAD DATASET
# ================================================================
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n"
        "Please check the INPUT_FILE path."
    )

data = pd.read_csv(INPUT_FILE)
print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)

df = data.copy()

# ================================================================
# 3. BASIC DATA INFORMATION
# ================================================================
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())

# ================================================================
# 4. AUTOMATIC TARGET COLUMN DETECTION
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
for col in target_candidates:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    target_column = df.columns[-1]
    print(
        "\nTarget column was not found using standard names."
        f"\nUsing last column as target: {target_column}"
    )

print("\nTarget column:", target_column)

# ================================================================
# 5. REMOVE ROWS WITH MISSING TARGET VALUES
# ================================================================
df = df.dropna(subset=[target_column]).copy()
X = df.drop(columns=[target_column])
y = df[target_column]

# ================================================================
# 6. ENCODE TARGET
# ================================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

print("\nTarget classes:")
for i, class_name in enumerate(label_encoder.classes_):
    print(i, "=", class_name)
number_of_classes = len(label_encoder.classes_)

# ================================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ================================================================
numeric_features = X.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)

# ================================================================
# 8. PREPROCESSING
# ================================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", __import__("sklearn").preprocessing.OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

transformers = []
if numeric_features:
    transformers.append(("num", numeric_transformer, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)

# ================================================================
# 9. TRAIN-TEST SPLIT
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
# 10. GRADIENT BOOSTING MODEL
# ================================================================
gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.10,
    max_depth=3,
    random_state=42
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", gradient_boosting)
])

# ================================================================
# 11. TRAIN MODEL
# ================================================================
print("\nTraining Gradient Boosting model...")
model.fit(X_train, y_train)
print("Training completed successfully.")

# ================================================================
# 12. PREDICTION
# ================================================================
y_pred = model.predict(X_test)
if hasattr(model, "predict_proba"):
    y_probability = model.predict_proba(X_test)
else:
    y_probability = None

# ================================================================
# 13. PERFORMANCE METRICS
# ================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ================================================================
# 14. SAVE PERFORMANCE RESULTS
# ================================================================
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})
metrics_file = os.path.join(OUTPUT_FOLDER, "gradient_boosting_metrics.csv")
metrics_df.to_csv(metrics_file, index=False)

# ================================================================
# 15. CLASSIFICATION REPORT
# ================================================================
report = classification_report(
    y_test,
    y_pred,
    target_names=[str(c) for c in label_encoder.classes_],
    zero_division=0
)
print("\nClassification Report:\n", report)

report_file = os.path.join(OUTPUT_FOLDER, "classification_report.txt")
with open(report_file, "w", encoding="utf-8") as file:
    file.write("GRADIENT BOOSTING CLASSIFICATION REPORT\n")
    file.write("=" * 60 + "\n\n")
    file.write(report)

# ================================================================
# 16. SAVE ACTUAL VS PREDICTED RESULTS
# ================================================================
actual_labels = label_encoder.inverse_transform(y_test)
predicted_labels = label_encoder.inverse_transform(y_pred)
prediction_results = pd.DataFrame({
    "Actual": actual_labels,
    "Predicted": predicted_labels,
    "Correct": actual_labels == predicted_labels
})

if y_probability is not None:
    for i, class_name in enumerate(label_encoder.classes_):
        prediction_results["Probability_" + str(class_name)] = y_probability[:, i]

prediction_file = os.path.join(OUTPUT_FOLDER, "gradient_boosting_predictions.csv")
prediction_results.to_csv(prediction_file, index=False)

# ================================================================
# 17. CONFUSION MATRIX
# ================================================================
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
cm_file = os.path.join(OUTPUT_FOLDER, "confusion_matrix.csv")
cm_df.to_csv(cm_file)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues", values_format="d")
plt.title("Gradient Boosting - Confusion Matrix")
plt.tight_layout()
cm_plot = os.path.join(OUTPUT_FOLDER, "confusion_matrix.png")
plt.savefig(cm_plot, dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
# 18. FEATURE IMPORTANCE
# ================================================================
preprocessor_fitted = model.named_steps["preprocessor"]
gb_model = model.named_steps["classifier"]
try:
    feature_names = preprocessor_fitted.get_feature_names_out()
    importances = gb_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    feature_importance_file = os.path.join(OUTPUT_FOLDER, "gradient_boosting_feature_importance.csv")
    feature_importance_df.to_csv(feature_importance_file, index=False)

    top_n = min(20, len(feature_importance_df))
    top_features = feature_importance_df.head(top_n)
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Gradient Boosting - Top Feature Importance")
    plt.tight_layout()
    importance_plot = os.path.join(OUTPUT_FOLDER, "gradient_boosting_feature_importance.png")
    plt.savefig(importance_plot, dpi=300, bbox_inches="tight")
    plt.close()
except Exception as e:
    print("\nFeature importance could not be generated:", e)

# ================================================================
# 20. ROC CURVE
# ================================================================
try:
    if y_probability is not None:
        plt.figure(figsize=(8, 6))
        if number_of_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_probability[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        else:
            for i in range(number_of_classes):
                binary_true = (y_test == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_true, y_probability[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Gradient Boosting - ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_file = os.path.join(OUTPUT_FOLDER, "gradient_boosting_roc_curve.png")
        plt.savefig(roc_file, dpi=300, bbox_inches="tight")
        plt.close()
except Exception as e:
    print("\nROC curve could not be generated:", e)

# ================================================================
# 21. MODEL PARAMETERS
# ================================================================
parameters = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Number of Estimators",
        "Learning Rate",
        "Maximum Depth",
        "Test Size",
        "Random State"
    ],
    "Value": [
        "Gradient Boosting Classifier",
        100,
        0.10,
        3,
        0.20,
        42
    ]
})
parameters_file = os.path.join(OUTPUT_FOLDER, "gradient_boosting_parameters.csv")
parameters.to_csv(parameters_file, index=False)

# ================================================================
# 22. DATASET INFORMATION
# ================================================================
dataset_info = pd.DataFrame({
    "Information": [
        "Original Rows",
        "Original Columns",
        "Rows Used",
        "Number of Features",
        "Training Samples",
        "Testing Samples",
        "Target Column",
        "Number of Classes"
    ],
    "Value": [
        data.shape[0],
        data.shape[1],
        df.shape[0],
        X.shape[1],
        X_train.shape[0],
        X_test.shape[0],
        target_column,
        number_of_classes
    ]
})
dataset_info_file = os.path.join(OUTPUT_FOLDER, "dataset_information.csv")
dataset_info.to_csv(dataset_info_file, index=False)

# ================================================================
# 23. FINAL OUTPUT SUMMARY
# ================================================================
print("\n" + "=" * 70)
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print("=" * 70)
print("\nOutput folder:", OUTPUT_FOLDER)
print("\nGenerated files:")
for file_name in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", file_name)
print("\n" + "=" * 70)
print("ORIGINAL DATASET WAS NOT MODIFIED.")
print("=" * 70)
