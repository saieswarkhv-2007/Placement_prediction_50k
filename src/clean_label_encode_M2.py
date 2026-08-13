import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# Load Placement Prediction Dataset
# Original dataset will NOT be modified
# ==========================================================


df = pd.read_csv("/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/placement_predict_50K_Raw.csv")


# Create a copy for processing
data = df.copy()


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================


for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values
# ==========================================================


print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================


before = data.shape[0]


data = data.drop_duplicates()


after = data.shape[0]


print("\nDuplicate Records Removed:",
      before - after)


# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================


num_cols = data.select_dtypes(
    include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
    exclude=np.number
).columns.tolist()


# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================


if len(num_cols) > 0:


    num_imputer = SimpleImputer(
        strategy="mean"
    )


    data[num_cols] = num_imputer.fit_transform(
        data[num_cols]
    )


# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================


if len(cat_cols) > 0:


    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )


    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )


# ==========================================================
# 6. Label Encoding
# ==========================================================


label_encoders = {}


for col in cat_cols:


    encoder = LabelEncoder()


    data[col] = encoder.fit_transform(
        data[col]
    )


    label_encoders[col] = encoder


# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================


print("\nMissing Values After Cleaning:")
print(data.isnull().sum())


# ==========================================================
# Save Result
# ==========================================================


data.to_csv(
    "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/clean_label_encode_M2.csv",
    index=False
)


print("\n======================================")
print("Original dataset is NOT modified.")
print("Label Encoding completed successfully.")
print("Output file:")
print("clean_label_encode_M2.csv")
print("======================================")
