# ============================================================
# Decision Tree Classifier with Cross Validation
# Placement Prediction Preprocessed Dataset
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATH SETTINGS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "CV_DT_All_Classifi_Regres_Metrics_outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD PREPROCESSED DATASET
# ============================================================
print("=" * 70)
print("DECISION TREE CLASSIFIER - PLACEMENT PREDICTION")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nDataset not found:\n{INPUT_FILE}\n"
        "Please check the INPUT_FILE path."
    )

data = pd.read_csv(INPUT_FILE)
print("\nDataset loaded successfully.")
print("Dataset shape:", data.shape)

# ============================================================
# 3. REMOVE DUPLICATES FROM WORKING COPY ONLY
# ============================================================
# Original dataset is NEVER modified.
working_data = data.copy()
duplicate_count = working_data.duplicated().sum()
if duplicate_count > 0:
    working_data = working_data.drop_duplicates().reset_index(drop=True)

print("Duplicate rows found:", duplicate_count)
print("Working dataset shape:", working_data.shape)

# ============================================================
# 4. AUTOMATIC TARGET COLUMN DETECTION
# ============================================================
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
    if col in working_data.columns:
        target_column = col
        break

# If standard target name is not found, use the last column as target.
if target_column is None:
    target_column = working_data.columns[-1]
    print(
        "\nTarget column was not found using standard names."
        f"\nUsing last column as target: {target_column}"
    )
else:
    print("\nTarget column detected:", target_column)

# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================
X = working_data.drop(columns=[target_column]).copy()
y = working_data[target_column].copy()

# ============================================================
# 6. HANDLE FEATURE DATA
# ============================================================
# Convert categorical feature columns into numeric columns.
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if len(categorical_columns) > 0:
    print("\nCategorical feature columns detected:")
    print(categorical_columns)
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False
    )

# Convert all feature values to numeric
X = X.apply(pd.to_numeric, errors="coerce")

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Fill missing values using median
for column in X.columns:
    if X[column].isna().any():
        median_value = X[column].median()
        if pd.isna(median_value):
            median_value = 0
        X[column] = X[column].fillna(median_value)

# ============================================================
# 7. ENCODE TARGET
# ============================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

print("\nTarget classes:")
for index, class_name in enumerate(label_encoder.classes_):
    print(index, "=", class_name)

# ============================================================
# 8. CHECK NUMBER OF CLASSES
# ============================================================
number_of_classes = len(np.unique(y_encoded))
if number_of_classes < 2:
    raise ValueError(
        "The target column contains fewer than two classes."
    )

print("\nNumber of target classes:", number_of_classes)

# ============================================================
# 9. SAVE DATASET INFORMATION
# ============================================================
dataset_info = pd.DataFrame({
    "Parameter": [
        "Original Rows",
        "Original Columns",
        "Working Rows",
        "Working Columns",
        "Target Column",
        "Number of Classes",
        "Number of Features",
        "Duplicate Rows Removed"
    ],
    "Value": [
        data.shape[0],
        data.shape[1],
        working_data.shape[0],
        working_data.shape[1],
        target_column,
        number_of_classes,
        X.shape[1],
        duplicate_count
    ]
})

dataset_info.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "dataset_information.csv"
    ),
    index=False
)

# ============================================================
# 10. DECISION TREE MODEL
# ============================================================
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
print("\nDecision Tree model created.")
print("Criterion: entropy")

# ============================================================
# 11. STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================
N_SPLITS = 5
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)
print("\nPerforming", N_SPLITS, "-Fold Stratified Cross Validation...")

# ============================================================
# 12. CROSS-VALIDATED PREDICTIONS
# ============================================================
y_pred = cross_val_predict(
    model,
    X,
    y_encoded,
    cv=cv,
    method="predict"
)

# ============================================================
# 13. CROSS-VALIDATED PROBABILITY PREDICTIONS
# ============================================================
y_probability = cross_val_predict(
    model,
    X,
    y_encoded,
    cv=cv,
    method="predict_proba"
)

# ============================================================
# 14. PRECISION, RECALL AND F1 SCORE
# ============================================================
precision = precision_score(
    y_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)
recall = recall_score(
    y_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)
f1 = f1_score(
    y_encoded,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# 15. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(
    y_encoded,
    y_pred
)
print("\nConfusion Matrix:")
print(cm)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{c}"
        for c in label_encoder.classes_
    ],
    columns=[
        f"Predicted_{c}"
        for c in label_encoder.classes_
    ]
)
cm_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.csv"
    )
)

# ============================================================
# 16. CLASSIFICATION REPORT
# ============================================================
classification_rep = classification_report(
    y_encoded,
    y_pred,
    target_names=[
        str(c)
        for c in label_encoder.classes_
    ],
    zero_division=0
)
print("\nClassification Report:")
print(classification_rep)

with open(
    os.path.join(
        OUTPUT_FOLDER,
        "classification_report.txt"
    ),
    "w"
) as file:
    file.write(classification_rep)

# ============================================================
# 17. ROC-AUC
# ============================================================
try:
    if number_of_classes == 2:
        roc_auc = roc_auc_score(
            y_encoded,
            y_probability[:, 1]
        )
        fpr, tpr, _ = roc_curve(
            y_encoded,
            y_probability[:, 1]
        )
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr,
            tpr,
            label=f"Decision Tree ROC-AUC = {roc_auc:.4f}"
        )
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Decision Tree ROC Curve - Cross Validation")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                "ROC_AUC_Curve.png"
            ),
            dpi=300
        )
        plt.close()
    else:
        roc_auc = roc_auc_score(
            y_encoded,
            y_probability,
            multi_class="ovr",
            average="weighted"
        )
except Exception as e:
    print("\nROC-AUC could not be calculated:", e)
    roc_auc = np.nan

# ============================================================
# 18. PR-AUC
# ============================================================
try:
    if number_of_classes == 2:
        pr_auc = average_precision_score(
            y_encoded,
            y_probability[:, 1]
        )
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_encoded,
            y_probability[:, 1]
        )
        plt.figure(figsize=(8, 6))
        plt.plot(
            recall_curve,
            precision_curve,
            label=f"PR-AUC = {pr_auc:.4f}"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Decision Tree Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                OUTPUT_FOLDER,
                "PR_AUC_Curve.png"
            ),
            dpi=300
        )
        plt.close()
    else:
        pr_auc = average_precision_score(
            pd.get_dummies(y_encoded),
            y_probability,
            average="weighted"
        )
except Exception as e:
    print("\nPR-AUC could not be calculated:", e)
    pr_auc = np.nan

# ============================================================
# 19. RMSE
# ============================================================
rmse = np.sqrt(
    mean_squared_error(
        y_encoded,
        y_pred
    )
)

# ============================================================
# 20. MAE
# ============================================================
mae = mean_absolute_error(
    y_encoded,
    y_pred
)

# ============================================================
# 21. MAPE
# ============================================================
non_zero_mask = y_encoded != 0
if np.any(non_zero_mask):
    mape = np.mean(
        np.abs(
            (
                y_encoded[non_zero_mask] - y_pred[non_zero_mask]
            )
            / y_encoded[non_zero_mask]
        )
    ) * 100
else:
    mape = np.nan

# ============================================================
# 22. P-SQUARE / R-SQUARED
# ============================================================
p_square = r2_score(
    y_encoded,
    y_pred
)

# ============================================================
# 23. OUTLIER SENSITIVITY ANALYSIS
# ============================================================
print("\nPerforming outlier sensitivity analysis...")
# IQR-based outlier detection on numerical features
outlier_mask = np.zeros(len(X), dtype=bool)
numeric_columns = X.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    Q1 = X[column].quantile(0.25)
    Q3 = X[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    column_outlier = (
        (X[column] < lower_bound)
        | (X[column] > upper_bound)
    )
    outlier_mask = outlier_mask | column_outlier.values

outlier_count = np.sum(outlier_mask)
normal_count = np.sum(~outlier_mask)

# Predictions for outliers and normal observations
if outlier_count > 0:
    outlier_precision = precision_score(
        y_encoded[outlier_mask],
        y_pred[outlier_mask],
        average="weighted",
        zero_division=0
    )
    outlier_recall = recall_score(
        y_encoded[outlier_mask],
        y_pred[outlier_mask],
        average="weighted",
        zero_division=0
    )
    outlier_f1 = f1_score(
        y_encoded[outlier_mask],
        y_pred[outlier_mask],
        average="weighted",
        zero_division=0
    )
else:
    outlier_precision = np.nan
    outlier_recall = np.nan
    outlier_f1 = np.nan

if normal_count > 0:
    normal_precision = precision_score(
        y_encoded[~outlier_mask],
        y_pred[~outlier_mask],
        average="weighted",
        zero_division=0
    )
    normal_recall = recall_score(
        y_encoded[~outlier_mask],
        y_pred[~outlier_mask],
        average="weighted",
        zero_division=0
    )
    normal_f1 = f1_score(
        y_encoded[~outlier_mask],
        y_pred[~outlier_mask],
        average="weighted",
        zero_division=0
    )
else:
    normal_precision = np.nan
    normal_recall = np.nan
    normal_f1 = np.nan

# Sensitivity difference
if not np.isnan(outlier_f1) and not np.isnan(normal_f1):
    outlier_sensitivity_difference = (
        normal_f1 - outlier_f1
    )
else:
    outlier_sensitivity_difference = np.nan

outlier_results = pd.DataFrame({
    "Measure": [
        "Total Observations",
        "Outlier Observations",
        "Normal Observations",
        "Outlier Percentage",
        "Outlier Precision",
        "Outlier Recall",
        "Outlier F1",
        "Normal Precision",
        "Normal Recall",
        "Normal F1",
        "Outlier Sensitivity Difference"
    ],
    "Value": [
        len(X),
        outlier_count,
        normal_count,
        (outlier_count / len(X)) * 100,
        outlier_precision,
        outlier_recall,
        outlier_f1,
        normal_precision,
        normal_recall,
        normal_f1,
        outlier_sensitivity_difference
    ]
})

outlier_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "outlier_sensitivity.csv"
    ),
    index=False
)

# ============================================================
# 24. SAVE CROSS-VALIDATED PREDICTIONS
# ============================================================
prediction_output = working_data.copy()
prediction_output["Actual_Encoded"] = y_encoded
prediction_output["Predicted_Encoded"] = y_pred
prediction_output["Actual_Label"] = label_encoder.inverse_transform(y_encoded)
prediction_output["Predicted_Label"] = label_encoder.inverse_transform(y_pred)
prediction_output["Correct_Prediction"] = (y_encoded == y_pred)

if number_of_classes == 2:
    prediction_output["Prediction_Probability"] = y_probability[:, 1]

prediction_output.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "cross_validated_predictions.csv"
    ),
    index=False
)

# ============================================================
# 25. TRAIN FINAL MODEL FOR FEATURE IMPORTANCE
# ============================================================
model.fit(X, y_encoded)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)
feature_importance.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "feature_importance.csv"
    ),
    index=False
)

# ============================================================
# 26. FEATURE IMPORTANCE GRAPH
# ============================================================
top_n = min(20, len(feature_importance))
top_features = feature_importance.head(top_n)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Decision Tree Feature Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "feature_importance.png"
    ),
    dpi=300
)
plt.close()

# ============================================================
# 27. CONFUSION MATRIX GRAPH
# ============================================================
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(
    range(number_of_classes),
    label_encoder.classes_,
    rotation=45
)
plt.yticks(
    range(number_of_classes),
    label_encoder.classes_
)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )
plt.colorbar()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=300
)
plt.close()

# ============================================================
# 28. OUTLIER SENSITIVITY GRAPH
# ============================================================
comparison_labels = [
    "Normal F1",
    "Outlier F1"
]
comparison_values = [
    normal_f1,
    outlier_f1
]

if not np.isnan(outlier_f1):
    plt.figure(figsize=(7, 5))
    plt.bar(
        comparison_labels,
        comparison_values
    )
    plt.ylabel("F1 Score")
    plt.title("Decision Tree Outlier Sensitivity")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "outlier_sensitivity.png"
        ),
        dpi=300
    )
    plt.close()

# ============================================================
# 29. PERFORMANCE RESULTS
# ============================================================
performance_results = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "RMSE",
        "MAE",
        "MAPE (%)",
        "P-Square (R2)",
        "Outlier Precision",
        "Outlier Recall",
        "Outlier F1",
        "Normal Precision",
        "Normal Recall",
        "Normal F1",
        "Outlier Sensitivity Difference"
    ],
    "Score": [
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        rmse,
        mae,
        mape,
        p_square,
        outlier_precision,
        outlier_recall,
        outlier_f1,
        normal_precision,
        normal_recall,
        normal_f1,
        outlier_sensitivity_difference
    ]
})
performance_results.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "performance_results.csv"
    ),
    index=False
)

# ============================================================
# 30. MODEL DETAILS
# ============================================================
model_details = pd.DataFrame({
    "Parameter": [
        "Model",
        "Criterion",
        "Cross Validation",
        "Number of Folds",
        "Shuffle",
        "Random State",
        "Number of Features",
        "Number of Classes"
    ],
    "Value": [
        "Decision Tree Classifier",
        "Entropy",
        "Stratified K-Fold",
        N_SPLITS,
        True,
        42,
        X.shape[1],
        number_of_classes
    ]
})
model_details.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "model_details.csv"
    ),
    index=False
)

# ============================================================
# 31. PRINT FINAL RESULTS
# ============================================================
print("\n")
print("=" * 70)
print("FINAL PERFORMANCE RESULTS")
print("=" * 70)
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")
if not np.isnan(mape):
    print(f"MAPE      : {mape:.4f}%")
else:
    print("MAPE      : Not applicable")
print(f"P-Square (R²) : {p_square:.4f}")

print("\nOutlier Sensitivity")
print(f"Outlier Count : {outlier_count}")
print(f"Normal Count  : {normal_count}")
if not np.isnan(outlier_f1):
    print(f"Outlier F1 : {outlier_f1:.4f}")
if not np.isnan(normal_f1):
    print(f"Normal F1  : {normal_f1:.4f}")

print("\nResults saved to:")
print(OUTPUT_FOLDER)
print("\nFiles generated:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)

print("\n")
print("=" * 70)
print("PROGRAM COMPLETED SUCCESSFULLY")
print("=" * 70)
