# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# Missing Values Handled Using Median Imputation
# Images are stored in ONE separate folder
# ============================================================


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = (
    "/Users/khvsaieswar/Desktop/"
    "Placement_prediction/dataset/"
    "final_preprocess_M2.csv"
)


if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )


data = pd.read_csv(
    DATASET_PATH
)


print("=" * 60)
print("LINEAR REGRESSION")
print("CLOSED-FORM NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 60)


print("\nDataset shape:")
print(data.shape)


print("\nFirst 5 rows:")
print(data.head())


# ============================================================
# 2. DISPLAY COLUMNS
# ============================================================

print("\nDataset columns:")

for column in data.columns:

    print(column)


# ============================================================
# 3. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = (
    "/Users/khvsaieswar/Desktop/"
    "Placement_prediction/outputs/"
    "Linear_Regression_CFNE_GD_Compare_M2"
)


os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)


print("\nImage output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 4. EXTRACT FEATURES AND TARGET
# ============================================================

# All columns except the last column = features
#
# Last column = target
#
# In your dataset:
#
# PlacementStatus is the last column.


X = data.iloc[:, :-1].copy()

y = data.iloc[:, -1].copy()


print("\nNumber of features:")
print(X.shape[1])


print("\nTarget column:")
print(data.columns[-1])


# ============================================================
# 5. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPE CHECK")
print("=" * 60)


print(
    X.dtypes
)


# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUE CHECK")
print("=" * 60)


print("\nMissing values in each feature:")

missing_features = X.isna().sum()

print(
    missing_features
)


total_feature_missing = (
    X.isna().sum().sum()
)


print(
    "\nTotal missing feature values:"
)

print(
    total_feature_missing
)


print(
    "\nMissing target values:"
)

print(
    y.isna().sum()
)


# ============================================================
# 7. HANDLE MISSING TARGET VALUES
# ============================================================

# We cannot train a supervised learning model
# if the target value is missing.
#
# Therefore, rows with missing target values
# are removed.


target_missing = y.isna().sum()


if target_missing > 0:

    print(
        "\nRemoving rows with missing target values..."
    )

    valid_target_rows = y.notna()

    X = X.loc[
        valid_target_rows
    ].copy()

    y = y.loc[
        valid_target_rows
    ].copy()


print(
    "\nDataset shape after removing "
    "missing target values:"
)

print(
    X.shape
)


# ============================================================
# 8. MAKE SURE ALL FEATURES ARE NUMERIC
# ============================================================

# Your final_preprocess_M2.csv should already contain
# numeric values.
#
# This conversion ensures that values are treated
# numerically by NumPy and sklearn.


for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


y = pd.to_numeric(
    y,
    errors="coerce"
)


# ============================================================
# 9. HANDLE TARGET VALUES THAT BECAME NaN
# ============================================================

# If any non-numeric target values were converted
# into NaN, remove those rows.


if y.isna().sum() > 0:

    print(
        "\nAdditional invalid target values found."
    )

    print(
        "Removing those rows..."
    )

    valid_target_rows = y.notna()

    X = X.loc[
        valid_target_rows
    ].copy()

    y = y.loc[
        valid_target_rows
    ].copy()


# ============================================================
# 10. FINAL MISSING VALUE CHECK BEFORE SPLIT
# ============================================================

print("\n" + "=" * 60)
print("FINAL MISSING VALUE CHECK")
print("=" * 60)


print(
    "\nMissing feature values:"
)

print(
    X.isna().sum().sum()
)


print(
    "\nMissing target values:"
)

print(
    y.isna().sum()
)


# ============================================================
# 11. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)


print(
    "\nTraining samples:"
)

print(
    len(X_train)
)


print(
    "\nTesting samples:"
)

print(
    len(X_test)
)


# ============================================================
# 12. MISSING VALUE IMPUTATION
# ============================================================

# Median imputation is used.
#
# IMPORTANT:
#
# The imputer is fitted ONLY on the training dataset.
#
# The same fitted imputer is then used on the test dataset.
#
# This prevents data leakage.


imputer = SimpleImputer(
    strategy="median"
)


X_train_imputed = imputer.fit_transform(
    X_train
)


X_test_imputed = imputer.transform(
    X_test
)


# Convert back to DataFrames


X_train = pd.DataFrame(

    X_train_imputed,

    columns=X_train.columns,

    index=X_train.index
)


X_test = pd.DataFrame(

    X_test_imputed,

    columns=X_test.columns,

    index=X_test.index
)


# ============================================================
# 13. CHECK NaN AFTER IMPUTATION
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES AFTER IMPUTATION")
print("=" * 60)


print(
    "\nTraining data NaN count:"
)

print(
    X_train.isna().sum().sum()
)


print(
    "\nTesting data NaN count:"
)

print(
    X_test.isna().sum().sum()
)


# ============================================================
# 14. CONVERT TO NUMPY ARRAYS
# ============================================================

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values


# ============================================================
# 15. FEATURE SCALING
#     IMPORTANT FOR GRADIENT DESCENT
# ============================================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 16. CLOSED FORM SOLUTION
#     NORMAL EQUATION
# ============================================================

print("\n" + "=" * 60)
print("CLOSED-FORM NORMAL EQUATION")
print("=" * 60)


# ------------------------------------------------------------
# Add bias/intercept column
# ------------------------------------------------------------

X_train_bias = np.c_[

    np.ones(
        (X_train.shape[0], 1)
    ),

    X_train

]


X_test_bias = np.c_[

    np.ones(
        (X_test.shape[0], 1)
    ),

    X_test

]


# ------------------------------------------------------------
# Normal Equation
#
# theta = (X^T X)^(-1) X^T y
#
# Instead of np.linalg.inv(), we use pinv().
#
# pinv() is safer when X^T X is singular
# or nearly singular.
# ------------------------------------------------------------


theta = np.linalg.pinv(
    X_train_bias
).dot(
    y_train
)


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

pred_normal = X_test_bias.dot(
    theta
)


# ============================================================
# 17. NORMAL EQUATION METRICS
# ============================================================

mse_normal = mean_squared_error(

    y_test,

    pred_normal

)


r2_normal = r2_score(

    y_test,

    pred_normal

)


print(
    "\nCoefficients:"
)

print(
    theta
)


print(
    "\nMSE:"
)

print(
    mse_normal
)


print(
    "\nR2 Score:"
)

print(
    r2_normal
)


# ============================================================
# 18. GRADIENT DESCENT
# ============================================================

print("\n" + "=" * 60)
print("GRADIENT DESCENT")
print("=" * 60)


# Add bias column to scaled data


X_train_gd = np.c_[

    np.ones(
        (X_train_scaled.shape[0], 1)
    ),

    X_train_scaled

]


X_test_gd = np.c_[

    np.ones(
        (X_test_scaled.shape[0], 1)
    ),

    X_test_scaled

]


# Number of training examples


m = len(
    y_train
)


# Initialize parameters


theta_gd = np.zeros(
    X_train_gd.shape[1]
)


# Learning rate


learning_rate = 0.01


# Number of iterations


epochs = 1000


# ============================================================
# 19. STORE LOSS FOR EACH EPOCH
# ============================================================

loss_history = []


# ============================================================
# 20. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(
    epochs
):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = X_train_gd.dot(
        theta_gd
    )


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    errors = (
        predictions
        -
        y_train
    )


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = (

        (2 / m)

        *

        X_train_gd.T.dot(
            errors
        )

    )


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    theta_gd -= (

        learning_rate
        *
        gradients

    )


    # --------------------------------------------------------
    # Calculate MSE
    # --------------------------------------------------------

    current_predictions = (
        X_train_gd.dot(
            theta_gd
        )
    )


    loss = np.mean(

        (
            current_predictions
            -
            y_train
        ) ** 2

    )


    loss_history.append(
        loss
    )


# ============================================================
# 21. GRADIENT DESCENT PREDICTION
# ============================================================

pred_gd = X_test_gd.dot(
    theta_gd
)


# ============================================================
# 22. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(

    y_test,

    pred_gd

)


r2_gd = r2_score(

    y_test,

    pred_gd

)


print(
    "\nCoefficients:"
)

print(
    theta_gd
)


print(
    "\nMSE:"
)

print(
    mse_gd
)


print(
    "\nR2 Score:"
)

print(
    r2_gd
)


# ============================================================
# 23. COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("NORMAL EQUATION VS GRADIENT DESCENT")
print("=" * 60)


print(
    "\nNormal Equation:"
)

print(
    f"MSE = {mse_normal:.6f}"
)

print(
    f"R2  = {r2_normal:.6f}"
)


print(
    "\nGradient Descent:"
)

print(
    f"MSE = {mse_gd:.6f}"
)

print(
    f"R2  = {r2_gd:.6f}"
)


# ============================================================
# 24. COMPARE RESULTS
# ============================================================

mse_difference = abs(
    mse_normal - mse_gd
)


r2_difference = abs(
    r2_normal - r2_gd
)


print(
    "\nDifference between methods:"
)


print(
    f"MSE difference = {mse_difference:.6f}"
)


print(
    f"R2 difference  = {r2_difference:.6f}"
)


# ============================================================
# IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_test,

    pred_normal,

    alpha=0.5,

    label="Normal Equation"

)


plt.scatter(

    y_test,

    pred_gd,

    alpha=0.5,

    label="Gradient Descent"

)


# ------------------------------------------------------------
# Perfect prediction line
# ------------------------------------------------------------

minimum = min(

    y_test.min(),

    pred_normal.min(),

    pred_gd.min()

)


maximum = max(

    y_test.max(),

    pred_normal.max(),

    pred_gd.max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--",

    label="Perfect Prediction"

)


plt.xlabel(
    "Actual Values"
)


plt.ylabel(
    "Predicted Values"
)


plt.title(
    "Actual vs Predicted Values"
)


plt.legend()


# Grid lines

plt.grid(
    True
)


plt.tight_layout()


image1 = os.path.join(

    IMAGE_FOLDER,

    "actual_vs_predicted.png"

)


plt.savefig(

    image1,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nImage saved:"
)

print(
    image1
)


# ============================================================
# IMAGE 2
# RESIDUAL COMPARISON
# ============================================================

# Residual = Actual - Predicted


normal_residuals = (

    y_test
    -
    pred_normal

)


gd_residuals = (

    y_test
    -
    pred_gd

)


plt.figure(
    figsize=(9, 6)
)


plt.scatter(

    pred_normal,

    normal_residuals,

    alpha=0.5,

    label="Normal Equation"

)


plt.scatter(

    pred_gd,

    gd_residuals,

    alpha=0.5,

    label="Gradient Descent"

)


# Zero residual line


plt.axhline(

    y=0,

    linestyle="--"

)


plt.xlabel(
    "Predicted Values"
)


plt.ylabel(
    "Residuals"
)


plt.title(
    "Residual Comparison"
)


plt.legend()


# Grid lines

plt.grid(
    True
)


plt.tight_layout()


image2 = os.path.join(

    IMAGE_FOLDER,

    "residual_comparison.png"

)


plt.savefig(

    image2,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nImage saved:"
)

print(
    image2
)


# ============================================================
# IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(
    figsize=(9, 6)
)


plt.plot(

    range(
        1,
        epochs + 1
    ),

    loss_history

)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Mean Squared Error"
)


plt.title(
    "Gradient Descent Convergence"
)


# Grid lines

plt.grid(
    True
)


plt.tight_layout()


image3 = os.path.join(

    IMAGE_FOLDER,

    "gradient_descent_loss.png"

)


plt.savefig(

    image3,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nImage saved:"
)

print(
    image3
)


# ============================================================
# 25. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({

    "Image": [

        "actual_vs_predicted.png",

        "residual_comparison.png",

        "gradient_descent_loss.png"

    ],

    "Description": [

        "Actual values versus predictions from both methods",

        "Residual comparison between Normal Equation and Gradient Descent",

        "MSE loss across Gradient Descent epochs"

    ]

})


image_info_file = os.path.join(

    IMAGE_FOLDER,

    "image_information.csv"

)


image_info.to_csv(

    image_info_file,

    index=False

)


print(
    "\nImage information saved:"
)

print(
    image_info_file
)


# ============================================================
# 26. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)


print(
    "\nOriginal dataset was NOT modified."
)


print(
    "\nAll images are stored in ONE folder:"
)


print(
    IMAGE_FOLDER
)


print(
    "\nGenerated images:"
)


print(
    "1. actual_vs_predicted.png"
)


print(
    "2. residual_comparison.png"
)


print(
    "3. gradient_descent_loss.png"
)


print(
    "\nGenerated information file:"
)


print(
    "4. image_information.csv"
)


print("\n" + "=" * 60)
print("ALL OUTPUTS SAVED SUCCESSFULLY")
print("=" * 60)