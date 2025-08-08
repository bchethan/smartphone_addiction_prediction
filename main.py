# ✅ STEP 1: IMPORT LIBRARIES
# Load everything needed for data handling, visualization, machine learning, and evaluation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

# ✅ STEP 2: LOAD & INSPECT DATA

df = pd.read_csv('phone_usage_india.csv')  # replace with your file

print("Shape of dataset:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# ✅ STEP 3: CLEANING & FEATURE ENGINEERING

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')

# Create target variable (1 = addicted if screen time > 8 hrs/day)
df['addiction_label'] = (df['Screen_Time_hrs_day'] > 8).astype(int)

# Label encode all categorical columns
df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])
df['Location_encoded'] = LabelEncoder().fit_transform(df['Location'])
df['Phone_Brand_encoded'] = LabelEncoder().fit_transform(df['Phone_Brand'])
df['OS_encoded'] = LabelEncoder().fit_transform(df['OS'])
df['Primary_Use_encoded'] = LabelEncoder().fit_transform(df['Primary_Use'])

print(f"\nAddiction distribution:\n{df['addiction_label'].value_counts()}")
print(f"Addiction rate: {df['addiction_label'].mean():.1%}")

# ✅ STEP 4: EDA (Exploratory Data Analysis)

# Plot screen time distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Screen_Time_hrs_day'], kde=True)
plt.title("Screen Time Distribution")
plt.xlabel("Screen Time (hours/day)")
plt.show()

# Correlation Heatmap (numeric columns only)
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Addiction rate pie chart
addiction_rate = df['addiction_label'].value_counts()
labels = ['Not Addicted', 'Addicted']
plt.figure(figsize=(8, 6))
plt.pie(addiction_rate, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Smartphone Addiction Rate")
plt.show()

# ✅ STEP 5: MACHINE LEARNING MODELS

# Select input features and target
features = ['Age', 'Gender_encoded', 'Location_encoded', 'Phone_Brand_encoded', 'OS_encoded',
            'Screen_Time_hrs_day', 'Data_Usage_GB_month', 'Calls_Duration_mins_day',
            'Number_of_Apps_Installed', 'Social_Media_Time_hrs_day', 'E-commerce_Spend_INR_month',
            'Streaming_Time_hrs_day', 'Gaming_Time_hrs_day', 'Monthly_Recharge_Cost_INR',
            'Primary_Use_encoded']
X = df[features]
y = df['addiction_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Try multiple models and compare
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Naive Bayes': GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ✅ STEP 6: COMPARE MODEL PERFORMANCE (Bar Chart)

results_df = pd.DataFrame(results).T
plt.figure(figsize=(10, 5))
results_df.plot(kind='bar', figsize=(10, 5), title='Model Performance Comparison')
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ STEP 7: FEATURE IMPORTANCE (from Random Forest)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(importances)

# Plot
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title("Top Features for Predicting Addiction")
plt.show()
