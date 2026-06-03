
# Import Libraries

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset

print("Loading Dataset...")

df = pd.read_csv("coffee_sales.csv")

# Dataset Overview

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data Cleaning

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("\nDataset Shape After Cleaning:")
print(df.shape)

# Feature Selection

features = ['hour_of_day', 'money', 'Time_of_Day', 'Weekday', 'Month_name']

target = 'coffee_name'

X = df[features]
y = df[target]

# Encode Target Labels

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nCoffee Classes:")
print(label_encoder.classes_)

# Feature Groups

numeric_features = ['hour_of_day', 'money']

categorical_features = ['Time_of_Day', 'Weekday', 'Month_name']

# Numerical Pipeline

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

# Categorical Pipeline

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combined Preprocessor

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

# Build SVM Pipeline

svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(probability=True))])

# Hyperparameter Tuning

print("\nTraining Model...")

param_grid = {
    'classifier__C': [1, 10, 50],
    'classifier__gamma': ['scale', 0.1],
    'classifier__kernel': ['rbf']
}

grid_search = GridSearchCV(estimator=svm_pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Model Training

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Cross Validation Accuracy: {grid_search.best_score_ * 100:.2f}%")

# Prediction

y_pred = best_model.predict(X_test)

# Accuracy Evaluation

accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

# Classification Report

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# Coffee Sales Distribution

plt.figure(figsize=(12, 6))

coffee_counts = df['coffee_name'].value_counts()

sns.barplot(x=coffee_counts.index, y=coffee_counts.values)

plt.title("Top Selling Coffee Types")
plt.xlabel("Coffee Type")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Peak Hour Analysis

plt.figure(figsize=(12, 6))

sns.countplot(x='hour_of_day', data=df)

plt.title("Transactions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Transaction Count")

plt.tight_layout()
plt.show()

# Revenue Distribution

plt.figure(figsize=(12, 6))

sns.histplot(df['money'], bins=30, kde=True)

plt.title("Revenue Distribution")
plt.xlabel("Money")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Correlation Heatmap

plt.figure(figsize=(8, 5))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()

# Sample Prediction

print("\nSample Prediction")

sample_data = pd.DataFrame({
    'hour_of_day': [9],
    'money': [220],
    'Time_of_Day': ['Morning'],
    'Weekday': ['Monday'],
    'Month_name': ['January']
})

prediction = best_model.predict(sample_data)

prediction_probability = best_model.predict_proba(sample_data)

predicted_coffee = label_encoder.inverse_transform(prediction)

confidence = np.max(prediction_probability) * 100

print("\nInput Transaction:")
print(sample_data)

print(f"\nPredicted Coffee Type: {predicted_coffee[0]}")

print(f"Prediction Confidence: {confidence:.2f}%")

print("\nStudy Completed.")
