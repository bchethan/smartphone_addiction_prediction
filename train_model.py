import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the data
print("Loading data...")
df = pd.read_csv('phone_usage_india.csv')

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')

# Create target variable (1 = addicted if screen time > 8 hrs/day)
df['addiction_label'] = (df['Screen_Time_hrs_day'] > 8).astype(int)

# Label encode categorical variables
print("Encoding categorical variables...")
df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])
df['Location_encoded'] = LabelEncoder().fit_transform(df['Location'])

# Select features for the model (matching the order expected by the Streamlit app)
features = [
    'Age', 'Gender_encoded', 'Location_encoded', 'Screen_Time_hrs_day',
    'Data_Usage_GB_month', 'Calls_Duration_mins_day', 'Number_of_Apps_Installed',
    'Social_Media_Time_hrs_day', 'Streaming_Time_hrs_day', 'Gaming_Time_hrs_day',
    'Monthly_Recharge_Cost_INR'
]

X = df[features]
y = df['addiction_label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 5 Most Important Features:")
print(feature_importance.head())

# Save the model
print("\nSaving model...")
joblib.dump(model, 'smartphone_addiction_model.pkl')
print("✅ Model saved as 'smartphone_addiction_model.pkl'")

# Print encoding information for reference
print(f"\nEncoding Information:")
print(f"Gender encoding: {dict(zip(df['Gender'].unique(), df['Gender_encoded'].unique()))}")
print(f"Location encoding: {dict(zip(df['Location'].unique(), df['Location_encoded'].unique()))}")

print(f"\nModel is ready to use with the Streamlit app!")
print(f"Run: streamlit run app.py")
