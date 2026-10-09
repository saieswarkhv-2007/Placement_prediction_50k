# ================================================================
# ADABOOST CLASSIFIER
# PLACEMENT PREDICTION USING RAW DATASET
# ================================================================
#
# IMPORTANT:
# The original RAW dataset is NEVER modified.
#
# All preprocessing is performed on copies / inside a pipeline.
#
# OUTPUTS:
# 1. Accuracy, Precision, Recall, F1 Score
# 2. Confusion Matrix
# 3. Performance Graph
# 4. Actual vs Predicted Chart
# 5. Class Distribution Chart
# 6. Feature Importance Chart
# 7. AdaBoost Base Tree Visualization
# 8. Classification Report
# 9. Prediction CSV
# 10. Trained Model & Parameters
# ================================================================
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ================================================================
# 1. PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "AdaBoost_Classifier_Outputs")

METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
PREDICTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "predictions")
CONFUSION_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrix")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
TREE_FOLDER = os.path.join(OUTPUT_FOLDER, "adaboost_tree")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "feature_importance")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")

folders = [
    OUTPUT_FOLDER,
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    TREE_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ================================================================
# 2. LOAD DATASET
# ================================================================
print("\n" + "=" * 80)
print(" ADABOOST PLACEMENT PREDICTION")
print("=" * 80)

if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found:", DATASET_PATH)
    raise SystemExit

df = pd.read_csv(DATASET_PATH)
print("\nRaw dataset loaded successfully.")
print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

data = df.copy()

possible_targets = ["PlacementStatus", "Placement", "Status", "Placed", "CGPA_Tier"]
target_column = None
for column in possible_targets:
    if column in data.columns:
        target_column = column
        break

if target_column is None:
    target_column = data.columns[-1]

print("\nTarget column:", target_column)

data_model = data.dropna(subset=[target_column]).copy()
X = data_model.drop(columns=[target_column]).copy()
y = data_model[target_column].copy()

empty_columns = X.columns[X.isnull().all()].tolist()
if len(empty_columns) > 0:
    X = X.drop(columns=empty_columns)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))

# ================================================================
# 3. CREATE ADABOOST MODEL
# ================================================================
base_tree = DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

try:
    adaboost = AdaBoostClassifier(
        estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )
except TypeError:
    adaboost = AdaBoostClassifier(
        base_estimator=base_tree,
        n_estimators=100,
        learning_rate=0.8,
        random_state=42
    )

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", adaboost)
])

print("\nTraining AdaBoost...")
model.fit(X_train, y_train)
print("AdaBoost training completed.")

# ================================================================
# 4. PREDICTION & METRICS
# ================================================================
print("\nGenerating predictions...")
y_pred = model.predict(X_test)
print("Prediction completed.")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\n" + "=" * 80)
print(" ADABOOST PERFORMANCE")
print("=" * 80)
print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1],
    "Percentage": [accuracy * 100, precision * 100, recall * 100, f1 * 100]
})
metrics_df.to_csv(os.path.join(METRICS_FOLDER, "adaboost_metrics.csv"), index=False)

# Classification Report
classification_report_result = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
classification_report_df = pd.DataFrame(classification_report_result).transpose()
classification_report_df.to_csv(os.path.join(METRICS_FOLDER, "classification_report.csv"))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
adaboost_classifier = model.named_steps["classifier"]
class_labels = adaboost_classifier.classes_

cm_df = pd.DataFrame(
    cm,
    index=["Actual_" + str(label) for label in class_labels],
    columns=["Predicted_" + str(label) for label in class_labels]
)
cm_df.to_csv(os.path.join(CONFUSION_FOLDER, "confusion_matrix.csv"))

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("AdaBoost - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(range(len(class_labels)), class_labels, rotation=45)
plt.yticks(range(len(class_labels)), class_labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(CONFUSION_FOLDER, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()

# Performance Graph
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values)
plt.title("AdaBoost Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "performance_graph.png"), dpi=300, bbox_inches="tight")
plt.close()

# Actual vs Predicted Chart
actual_counts = y_test.value_counts()
predicted_counts = pd.Series(y_pred).value_counts()
comparison_df = pd.DataFrame({"Actual": actual_counts, "Predicted": predicted_counts}).fillna(0)
comparison_df = comparison_df.reindex(class_labels)

plt.figure(figsize=(9, 6))
x = np.arange(len(class_labels))
width = 0.35
plt.bar(x - width / 2, comparison_df["Actual"], width, label="Actual")
plt.bar(x + width / 2, comparison_df["Predicted"], width, label="Predicted")
plt.xlabel("Placement Class")
plt.ylabel("Number of Students")
plt.title("Actual vs Predicted Placement - AdaBoost")
plt.xticks(x, class_labels)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
plt.close()

# Class Distribution Chart
class_counts = y.value_counts()
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.xlabel("Placement Class")
plt.ylabel("Number of Students")
plt.title("Placement Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_FOLDER, "class_distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

# Feature Importance
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
feature_importances = adaboost_classifier.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
feature_importance_df.to_csv(os.path.join(FEATURE_FOLDER, "feature_importance.csv"), index=False)

top_features = feature_importance_df.head(15).sort_values(by="Importance")
plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 AdaBoost Feature Importances")
plt.tight_layout()
plt.savefig(os.path.join(FEATURE_FOLDER, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()

# First Base Tree Visualization
first_tree = adaboost_classifier.estimators_[0]
plt.figure(figsize=(30, 18))
plot_tree(
    first_tree,
    feature_names=feature_names,
    class_names=[str(label) for label in class_labels],
    filled=True,
    rounded=True,
    proportion=False,
    precision=2,
    fontsize=8
)
plt.title("AdaBoost - First Base Decision Tree", fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(TREE_FOLDER, "adaboost_base_tree_1.png"), dpi=300, bbox_inches="tight")
plt.close()

# Save Test Predictions & Trained Model
test_predictions = X_test.copy()
test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred
test_predictions.to_csv(os.path.join(PREDICTIONS_FOLDER, "test_predictions.csv"), index=False)

with open(os.path.join(MODEL_FOLDER, "adaboost_model.pkl"), "wb") as file:
    pickle.dump(model, file)

parameters_df = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Number of Estimators",
        "Learning Rate",
        "Base Tree Maximum Depth",
        "Random State"
    ],
    "Value": [
        "AdaBoost Classifier",
        adaboost_classifier.n_estimators,
        adaboost_classifier.learning_rate,
        first_tree.max_depth,
        adaboost_classifier.random_state
    ]
})
parameters_df.to_csv(os.path.join(METRICS_FOLDER, "adaboost_parameters.csv"), index=False)

estimator_weights_df = pd.DataFrame({
    "Estimator_Number": np.arange(1, len(adaboost_classifier.estimator_weights_) + 1),
    "Estimator_Weight": adaboost_classifier.estimator_weights_
})
estimator_weights_df.to_csv(os.path.join(METRICS_FOLDER, "estimator_weights.csv"), index=False)

print("\n" + "=" * 80)
print(" ADABOOST COMPLETED SUCCESSFULLY")
print("=" * 80)
print("All outputs stored in:", OUTPUT_FOLDER)
