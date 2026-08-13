# --------------------------------------------
# Placement Prediction Dataset Preprocessing
# --------------------------------------------


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Read original dataset
input_file = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/placement_predict_50K_Raw.csv"
output_file = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/final_preprocess_M2.csv"


df = pd.read_csv(input_file)


# Create a copy so original dataset remains unchanged
processed_df = df.copy()


print("Original Dataset Shape:", processed_df.shape)


# --------------------------------------------
# Remove Duplicate Records
# --------------------------------------------
processed_df.drop_duplicates(inplace=True)


# --------------------------------------------
# Handle Missing Values
# --------------------------------------------


# Numeric Columns
numeric_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns


for col in numeric_cols:
    processed_df[col].fillna(processed_df[col].median(), inplace=True)


# Categorical Columns
categorical_cols = processed_df.select_dtypes(include=['object']).columns


for col in categorical_cols:
    processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)


# --------------------------------------------
# Clean Text Data
# --------------------------------------------
for col in categorical_cols:
    # strip() Removes leading and trailing spaces from text values
    processed_df[col] = processed_df[col].str.strip()
    # lower() converts all uppercase characters in a string into lowercase
    processed_df[col] = processed_df[col].str.lower()


# --------------------------------------------
# Label Encoding
# --------------------------------------------
encoder = LabelEncoder()


for col in categorical_cols:
    processed_df[col] = encoder.fit_transform(processed_df[col])


# --------------------------------------------
# Feature Scaling
# --------------------------------------------
scaler = StandardScaler()


processed_df[numeric_cols] = scaler.fit_transform(
    processed_df[numeric_cols]
)


# --------------------------------------------
# Save Preprocessed Dataset
# --------------------------------------------
processed_df.to_csv(output_file, index=False)


print("\nPreprocessing Completed Successfully!")
print("Original Dataset Shape :", df.shape)
print("Processed Dataset Shape:", processed_df.shape)
print("Saved File :", output_file)
