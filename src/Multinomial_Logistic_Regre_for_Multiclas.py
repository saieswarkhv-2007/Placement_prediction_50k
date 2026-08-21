# ==============================================================
# MULTINOMIAL LOGISTIC REGRESSION
# SOFTMAX REGRESSION FOR MULTI-CLASS PLACEMENT PREDICTION
#
# IMPORTANT:
# The original preprocessed dataset is NOT modified.
# All output files are saved inside the "output" folder.
# ==============================================================


import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import (
   accuracy_score,
   precision_score,
   recall_score,
   f1_score,
   log_loss,
   confusion_matrix,
   classification_report
)


warnings.filterwarnings("ignore")




# ==============================================================
# 1. FILE SETTINGS
# ==============================================================


# Your ORIGINAL preprocessed dataset
DATASET_FILE = "/Users/khvsaieswar/Desktop/Placement_prediction/dataset/final_preprocess_M2.csv"


# Target column
# CHANGE ONLY THIS NAME if your target column has another name.
TARGET_COLUMN = "CGPA_Tier"


# All generated files will be stored here
OUTPUT_FOLDER = "/Users/khvsaieswar/Desktop/Placement_prediction/outputs/Multinomial_Logistic_Regre_for_Multiclass_M2"




# ==============================================================
# 2. CREATE OUTPUT FOLDER
# ==============================================================


os.makedirs(OUTPUT_FOLDER, exist_ok=True)




print("=" * 75)
print("MULTINOMIAL LOGISTIC REGRESSION")
print("SOFTMAX REGRESSION - MULTI-CLASS PLACEMENT PREDICTION")
print("=" * 75)




# ==============================================================
# 3. CHECK DATASET
# ==============================================================


if not os.path.exists(DATASET_FILE):


   print("\nERROR: Dataset not found!")
   print("Expected file:", DATASET_FILE)
   print("\nPlace the CSV file in the same folder as main.py.")


   exit()




# ==============================================================
# 4. READ DATASET
# ==============================================================


# IMPORTANT:
# pd.read_csv() only READS the dataset.
# Nothing is written back to the original dataset.


df = pd.read_csv(DATASET_FILE)




print("\nDataset loaded successfully.")


print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])




# ==============================================================
# 5. KEEP ORIGINAL DATASET UNTOUCHED
# ==============================================================


# Create an in-memory copy.
# All further operations are performed on copies.


data = df.copy(deep=True)




# ==============================================================
# 6. DISPLAY DATASET
# ==============================================================


print("\n" + "=" * 75)
print("PREPROCESSED DATASET")
print("=" * 75)


print("\nFirst 5 rows:")
print(data.head())


print("\nColumn names:")
print(data.columns.tolist())


print("\nData types:")
print(data.dtypes)


print("\nMissing values:")
print(data.isnull().sum())




# ==============================================================
# 7. CHECK TARGET COLUMN
# ==============================================================


if TARGET_COLUMN not in data.columns:


   print("\nERROR:")
   print(
       f"Target column '{TARGET_COLUMN}' does not exist."
   )


   print("\nAvailable columns:")
   print(data.columns.tolist())


   exit()




print("\nTarget column:", TARGET_COLUMN)




# ==============================================================
# 8. CHECK NUMBER OF CLASSES
# ==============================================================


print("\n" + "=" * 75)
print("TARGET CLASS DISTRIBUTION")
print("=" * 75)


class_counts = data[TARGET_COLUMN].value_counts()


print(class_counts)


number_of_classes = data[TARGET_COLUMN].nunique()


print("\nNumber of classes:", number_of_classes)


if number_of_classes < 3:


   print(
       "\nWARNING: The target has fewer than 3 classes."
   )


   print(
       "This program is intended for a multi-class problem."
   )




# ==============================================================
# 9. SAVE DATASET INFORMATION
# ==============================================================
# This creates a NEW file.
# The original dataset is NOT changed.


information_file = os.path.join(
   OUTPUT_FOLDER,
   "dataset_information.txt"
)


with open(
   information_file,
   "w",
   encoding="utf-8"
) as file:


   file.write(
       "PREPROCESSED PLACEMENT DATASET INFORMATION\n"
   )


   file.write("=" * 75 + "\n\n")


   file.write(
       f"Dataset file: {DATASET_FILE}\n"
   )


   file.write(
       f"Rows: {data.shape[0]}\n"
   )


   file.write(
       f"Columns: {data.shape[1]}\n"
   )


   file.write(
       f"Target column: {TARGET_COLUMN}\n"
   )


   file.write(
       f"Number of classes: {number_of_classes}\n"
   )


   file.write("\nColumn names:\n")


   for column in data.columns:
       file.write(f"- {column}\n")


   file.write("\nData types:\n")
   file.write(str(data.dtypes))


   file.write("\n\nMissing values:\n")
   file.write(str(data.isnull().sum()))


   file.write("\n\nClass distribution:\n")
   file.write(str(class_counts))




# ==============================================================
# 10. SEPARATE FEATURES AND TARGET
# ==============================================================


# Use a COPY of the data.
# The original dataset remains unchanged.


X = data.drop(
   columns=[TARGET_COLUMN]
).copy()


y = data[TARGET_COLUMN].copy()




# ==============================================================
# 11. CHECK FEATURES
# ==============================================================


print("\n" + "=" * 75)
print("FEATURE INFORMATION")
print("=" * 75)


print("\nFeature columns:")


for column in X.columns:
   print("-", column)




print("\nNumber of features:", X.shape[1])




# ==============================================================
# 12. CHECK WHETHER FEATURES ARE NUMERICAL
# ==============================================================


non_numeric_columns = X.select_dtypes(
   exclude=[np.number]
).columns.tolist()




if len(non_numeric_columns) > 0:


   print("\nERROR:")
   print(
       "The dataset is expected to be already preprocessed."
   )


   print(
       "\nThese columns are still non-numerical:"
   )


   for column in non_numeric_columns:
       print("-", column)


   print(
       "\nPlease use the preprocessed dataset containing "
       "numerical features."
   )


   exit()




# ==============================================================
# 13. CHECK MISSING VALUES
# ==============================================================


missing_values = X.isnull().sum().sum()




if missing_values > 0:


   print("\nERROR:")
   print(
       "Missing values were found in the preprocessed features."
   )


   print("\nMissing values by column:")
   print(X.isnull().sum())


   print(
       "\nThe original dataset will NOT be modified."
   )


   print(
       "Please use a properly preprocessed dataset."
   )


   exit()




# ==============================================================
# 14. ENCODE ONLY THE TARGET
# ==============================================================


# IMPORTANT:
# We do NOT change the feature columns.
#
# LabelEncoder creates an internal numerical representation
# of the target classes for model training.
#
# This does NOT modify the original CSV file.


label_encoder = LabelEncoder()


y_encoded = label_encoder.fit_transform(y)




# ==============================================================
# 15. CLASS MAPPING
# ==============================================================


class_names = label_encoder.classes_


print("\n" + "=" * 75)
print("CLASS MAPPING")
print("=" * 75)


for number, class_name in enumerate(class_names):


   print(
       f"{number} --> {class_name}"
   )




# ==============================================================
# 16. TRAIN-TEST SPLIT
# ==============================================================


X_train, X_test, y_train, y_test = train_test_split(


   X,
   y_encoded,


   test_size=0.20,


   random_state=42,


   stratify=y_encoded
)




print("\n" + "=" * 75)
print("TRAIN-TEST SPLIT")
print("=" * 75)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


print(
   "\nTraining percentage:",
   round(
       len(X_train) / len(X) * 100,
       2
   ),
   "%"
)


print(
   "Testing percentage:",
   round(
       len(X_test) / len(X) * 100,
       2
   ),
   "%"
)




# ==============================================================
# 17. MULTINOMIAL LOGISTIC REGRESSION
# ==============================================================


print("\n" + "=" * 75)
print("MULTINOMIAL LOGISTIC REGRESSION")
print("=" * 75)


model = LogisticRegression(


   # Multinomial / Softmax regression
   solver="lbfgs",


   max_iter=2000,


   multi_class="multinomial",


   random_state=42
)




# ==============================================================
# 18. TRAIN MODEL
# ==============================================================


print("\nTraining model...")


model.fit(
   X_train,
   y_train
)


print("Model training completed successfully.")




# ==============================================================
# 19. PREDICT TEST DATA
# ==============================================================


y_pred = model.predict(
   X_test
)




# ==============================================================
# 20. SOFTMAX PROBABILITIES
# ==============================================================


# Probability assigned to every class


y_probability = model.predict_proba(
   X_test
)




# ==============================================================
# 21. CALCULATE ACCURACY
# ==============================================================


accuracy = accuracy_score(
   y_test,
   y_pred
)




# ==============================================================
# 22. CALCULATE PRECISION
# ==============================================================


precision = precision_score(


   y_test,


   y_pred,


   average="weighted",


   zero_division=0
)




# ==============================================================
# 23. CALCULATE RECALL
# ==============================================================


recall = recall_score(


   y_test,


   y_pred,


   average="weighted",


   zero_division=0
)




# ==============================================================
# 24. CALCULATE F1 SCORE
# ==============================================================


f1 = f1_score(


   y_test,


   y_pred,


   average="weighted",


   zero_division=0
)




# ==============================================================
# 25. CALCULATE CROSS-ENTROPY LOSS
# ==============================================================


cross_entropy = log_loss(


   y_test,


   y_probability,


   labels=np.arange(
       number_of_classes
   )
)




# ==============================================================
# 26. PRINT MODEL RESULTS
# ==============================================================


print("\n" + "=" * 75)
print("MODEL RESULTS")
print("=" * 75)


print(
   f"\nAccuracy             : {accuracy:.4f}"
)


print(
   f"Accuracy (%)         : {accuracy * 100:.2f}%"
)


print(
   f"Precision (Weighted) : {precision:.4f}"
)


print(
   f"Recall (Weighted)    : {recall:.4f}"
)


print(
   f"F1 Score (Weighted)  : {f1:.4f}"
)


print(
   f"Cross-Entropy Loss   : {cross_entropy:.4f}"
)




# ==============================================================
# 27. CLASSIFICATION REPORT
# ==============================================================


report = classification_report(


   y_test,


   y_pred,


   target_names=[
       str(x)
       for x in class_names
   ],


   zero_division=0
)




print("\n" + "=" * 75)
print("CLASSIFICATION REPORT")
print("=" * 75)


print(report)




# ==============================================================
# 28. SAVE MODEL METRICS
# ==============================================================


metrics_file = os.path.join(


   OUTPUT_FOLDER,


   "model_metrics.txt"
)




with open(


   metrics_file,


   "w",


   encoding="utf-8"


) as file:


   file.write(
       "MULTINOMIAL LOGISTIC REGRESSION RESULTS\n"
   )


   file.write("=" * 75 + "\n\n")


   file.write(
       f"Dataset: {DATASET_FILE}\n"
   )


   file.write(
       f"Target: {TARGET_COLUMN}\n"
   )


   file.write(
       f"Number of classes: {number_of_classes}\n\n"
   )


   file.write("CLASS MAPPING\n")
   file.write("-" * 40 + "\n")


   for number, class_name in enumerate(class_names):


       file.write(
           f"{number} = {class_name}\n"
       )


   file.write("\nMODEL PERFORMANCE\n")
   file.write("-" * 40 + "\n")


   file.write(
       f"Accuracy: {accuracy:.4f}\n"
   )


   file.write(
       f"Accuracy (%): {accuracy * 100:.2f}%\n"
   )


   file.write(
       f"Precision: {precision:.4f}\n"
   )


   file.write(
       f"Recall: {recall:.4f}\n"
   )


   file.write(
       f"F1 Score: {f1:.4f}\n"
   )


   file.write(
       f"Cross-Entropy Loss: {cross_entropy:.4f}\n"
   )


   file.write("\nCLASSIFICATION REPORT\n")
   file.write("-" * 40 + "\n")


   file.write(report)




# ==============================================================
# 29. CONFUSION MATRIX
# ==============================================================


cm = confusion_matrix(


   y_test,


   y_pred,


   labels=np.arange(
       number_of_classes
   )
)




plt.figure(
   figsize=(8, 6)
)




sns.heatmap(


   cm,


   annot=True,


   fmt="d",


   cmap="Blues",


   xticklabels=class_names,


   yticklabels=class_names
)




plt.title(
   "Confusion Matrix - Multinomial Logistic Regression"
)


plt.xlabel(
   "Predicted Class"
)


plt.ylabel(
   "Actual Class"
)


plt.tight_layout()




plt.savefig(


   os.path.join(
       OUTPUT_FOLDER,
       "confusion_matrix.png"
   ),


   dpi=300
)




plt.close()




# ==============================================================
# 30. CREATE PREDICTION OUTPUT
# ==============================================================


prediction_output = X_test.copy()




# Add actual class


prediction_output[
   "Actual_Class"
] = label_encoder.inverse_transform(
   y_test
)




# Add predicted class


prediction_output[
   "Predicted_Class"
] = label_encoder.inverse_transform(
   y_pred
)




# Add correct/incorrect


prediction_output[
   "Correct"
] = (
   y_test == y_pred
)




# ==============================================================
# 31. ADD SOFTMAX PROBABILITIES
# ==============================================================


for i, class_name in enumerate(class_names):


   probability_column = (
       "Probability_"
       + str(class_name)
       .replace(" ", "_")
   )


   prediction_output[
       probability_column
   ] = y_probability[:, i]




# Save predictions


prediction_output.to_csv(


   os.path.join(
       OUTPUT_FOLDER,
       "predictions.csv"
   ),


   index=False
)




# ==============================================================
# 32. SAVE SOFTMAX PROBABILITIES SEPARATELY
# ==============================================================


probability_output = pd.DataFrame(
   y_probability
)




probability_output.columns = [


   "Probability_"
   + str(class_name)
   .replace(" ", "_")


   for class_name in class_names


]




probability_output.to_csv(


   os.path.join(
       OUTPUT_FOLDER,
       "softmax_probabilities.csv"
   ),


   index=False
)




# ==============================================================
# 33. SAVE ACTUAL VS PREDICTED
# ==============================================================


actual_predicted = pd.DataFrame({


   "Actual_Class":
       label_encoder.inverse_transform(y_test),


   "Predicted_Class":
       label_encoder.inverse_transform(y_pred),


   "Correct":
       y_test == y_pred


})




actual_predicted.to_csv(


   os.path.join(
       OUTPUT_FOLDER,
       "actual_vs_predicted.csv"
   ),


   index=False
)




# ==============================================================
# 34. SAVE CLASS DISTRIBUTION
# ==============================================================


class_distribution = pd.DataFrame({


   "Class":
       class_names,


   "Count":
       [
           np.sum(
               y_encoded == i
           )


           for i in range(
               number_of_classes
           )
       ]


})




class_distribution.to_csv(


   os.path.join(
       OUTPUT_FOLDER,
       "class_distribution.csv"
   ),


   index=False
)




# ==============================================================
# 35. CLASS DISTRIBUTION GRAPH
# ==============================================================


plt.figure(
   figsize=(8, 5)
)




sns.barplot(


   data=class_distribution,


   x="Class",


   y="Count",


   hue="Class",


   legend=False
)




plt.title(
   "Placement Class Distribution"
)


plt.xlabel(
   "Placement Class"
)


plt.ylabel(
   "Number of Students"
)


plt.xticks(
   rotation=30
)


plt.tight_layout()




plt.savefig(


   os.path.join(
       OUTPUT_FOLDER,
       "class_distribution.png"
   ),


   dpi=300
)




plt.close()




# ==============================================================
# 36. SAVE MODEL
# ==============================================================


model_file = os.path.join(


   OUTPUT_FOLDER,


   "multinomial_logistic_regression.pkl"
)




joblib.dump(


   model,


   model_file
)




# ==============================================================
# 37. SAVE LABEL ENCODER
# ==============================================================


encoder_file = os.path.join(


   OUTPUT_FOLDER,


   "label_encoder.pkl"
)




joblib.dump(


   label_encoder,


   encoder_file
)




# ==============================================================
# 38. SAVE MODEL COEFFICIENTS
# ==============================================================


coefficients = pd.DataFrame(


   model.coef_,


   columns=X.columns,


   index=[
       str(x)
       for x in class_names
   ]
)




coefficients.to_csv(


   os.path.join(
       OUTPUT_FOLDER,
       "model_coefficients.csv"
   )
)




# ==============================================================
# 39. SAVE CLASS INTERCEPTS
# ==============================================================


intercepts = pd.DataFrame({


   "Class":
       [
           str(x)
           for x in class_names
       ],


   "Intercept":
       model.intercept_


})




intercepts.to_csv(


   os.path.join(
       OUTPUT_FOLDER,
       "model_intercepts.csv"
   ),


   index=False
)




# ==============================================================
# 40. FINAL MESSAGE
# ==============================================================


print("\n" + "=" * 75)
print("COMPLETED SUCCESSFULLY")
print("=" * 75)


print(
   "\nIMPORTANT:"
)


print(
   "Original preprocessed dataset was NOT modified."
)


print(
   "\nAll generated files are stored in:"
)


print(
   os.path.abspath(
       OUTPUT_FOLDER
   )
)




print("\nGenerated files:")


for filename in sorted(
   os.listdir(OUTPUT_FOLDER)
):


   print(
       "   ",
       filename
   )




print("\n" + "=" * 75)
