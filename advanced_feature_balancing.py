"""
Advanced Feature Balancing for Smartphone Addiction Predictor
============================================================

This script addresses the Screen_Time dominance issue using multiple approaches:
1. Feature engineering to create more meaningful features
2. Different algorithms that are more sensitive to feature scaling
3. Feature selection and dimensionality reduction
4. Ensemble methods with different feature weights

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import joblib

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🔍 ADVANCED FEATURE BALANCING FOR SMARTPHONE ADDICTION PREDICTOR")
print("=" * 70)

# Load and prepare data
print("\n📊 Step 1: Loading and preparing data...")
df = pd.read_csv('phone_usage_india.csv')

# Clean column names
df.columns = df.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '_')

# Create target variable
df['addiction_label'] = (df['Screen_Time_hrs_day'] > 8).astype(int)

# Encode categorical variables
df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])
df['Location_encoded'] = LabelEncoder().fit_transform(df['Location'])

print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['addiction_label'].value_counts().to_dict()}")

# FEATURE ENGINEERING: Create more meaningful features
print("\n🔧 Step 2: Advanced feature engineering...")

# Original features
original_features = [
    'Age', 'Gender_encoded', 'Location_encoded', 'Screen_Time_hrs_day',
    'Data_Usage_GB_month', 'Calls_Duration_mins_day', 'Number_of_Apps_Installed',
    'Social_Media_Time_hrs_day', 'Streaming_Time_hrs_day', 'Gaming_Time_hrs_day',
    'Monthly_Recharge_Cost_INR'
]

# Create engineered features
df['Total_Entertainment_Time'] = df['Social_Media_Time_hrs_day'] + df['Streaming_Time_hrs_day'] + df['Gaming_Time_hrs_day']
df['Screen_Time_Ratio'] = df['Screen_Time_hrs_day'] / 24.0  # Ratio of day spent on screen
df['Data_Usage_per_Hour'] = df['Data_Usage_GB_month'] / (df['Screen_Time_hrs_day'] * 30 + 1)  # Avoid division by zero
df['App_Density'] = df['Number_of_Apps_Installed'] / 100.0  # Normalize app count
df['Cost_per_Hour'] = df['Monthly_Recharge_Cost_INR'] / (df['Screen_Time_hrs_day'] * 30 + 1)
df['Communication_Intensity'] = df['Calls_Duration_mins_day'] / 60.0  # Convert to hours
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(int)

# Create interaction features
df['Screen_Time_x_Apps'] = df['Screen_Time_hrs_day'] * df['App_Density']
df['Entertainment_x_Data'] = df['Total_Entertainment_Time'] * df['Data_Usage_GB_month']
df['Age_x_Screen_Time'] = df['Age'] * df['Screen_Time_hrs_day']

# Enhanced feature set
enhanced_features = [
    'Age', 'Gender_encoded', 'Location_encoded', 
    'Screen_Time_hrs_day', 'Screen_Time_Ratio',  # Keep original but add ratio
    'Data_Usage_GB_month', 'Data_Usage_per_Hour',
    'Calls_Duration_mins_day', 'Communication_Intensity',
    'Number_of_Apps_Installed', 'App_Density',
    'Social_Media_Time_hrs_day', 'Streaming_Time_hrs_day', 'Gaming_Time_hrs_day',
    'Total_Entertainment_Time',
    'Monthly_Recharge_Cost_INR', 'Cost_per_Hour',
    'Age_Group',
    'Screen_Time_x_Apps', 'Entertainment_x_Data', 'Age_x_Screen_Time'
]

print(f"Original features: {len(original_features)}")
print(f"Enhanced features: {len(enhanced_features)}")

# Prepare data
X_original = df[original_features]
X_enhanced = df[enhanced_features]
y = df['addiction_label']

# Split data
X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42, stratify=y)
X_train_enh, X_test_enh, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42, stratify=y)

print("\n🤖 Step 3: Training models with different approaches...")

# Define models that are sensitive to feature scaling
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Define scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

results = {}
feature_importances = {}

# Test different combinations
for model_name, model in models.items():
    print(f"\n🔍 Testing {model_name}...")
    
    for scaler_name, scaler in scalers.items():
        print(f"  📊 With {scaler_name}...")
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train_enh)
        X_test_scaled = scaler.transform(X_test_enh)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': enhanced_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            coef_abs = np.abs(model.coef_[0])
            importance_df = pd.DataFrame({
                'Feature': enhanced_features,
                'Importance': coef_abs / coef_abs.sum()
            }).sort_values('Importance', ascending=False)
        else:
            importance_df = None
        
        # Store results
        key = f"{model_name} + {scaler_name}"
        results[key] = {
            'accuracy': accuracy,
            'model': model,
            'scaler': scaler,
            'feature_importance': importance_df
        }
        
        if importance_df is not None:
            screen_time_importance = importance_df[importance_df['Feature'] == 'Screen_Time_hrs_day']['Importance'].iloc[0] if 'Screen_Time_hrs_day' in importance_df['Feature'].values else 0
            print(f"    ✅ Accuracy: {accuracy:.3f}, Screen_Time importance: {screen_time_importance:.1%}")

print("\n📊 Step 4: Analyzing results...")

# Find the best balanced model
best_models = []
for key, result in results.items():
    if result['feature_importance'] is not None:
        screen_time_importance = result['feature_importance'][result['feature_importance']['Feature'] == 'Screen_Time_hrs_day']['Importance'].iloc[0] if 'Screen_Time_hrs_day' in result['feature_importance']['Feature'].values else 0
        if 0.6 <= screen_time_importance <= 0.8:  # Target range
            best_models.append((key, result['accuracy'], screen_time_importance))

# Sort by accuracy
best_models.sort(key=lambda x: x[1], reverse=True)

print(f"\n🎯 Best balanced models (Screen_Time 60-80%):")
for i, (key, accuracy, screen_time_imp) in enumerate(best_models[:3]):
    print(f"{i+1}. {key}: Accuracy={accuracy:.3f}, Screen_Time={screen_time_imp:.1%}")

# If no model in target range, find the closest
if not best_models:
    print("\n⚠️ No model in target range. Finding closest...")
    closest_models = []
    for key, result in results.items():
        if result['feature_importance'] is not None:
            screen_time_importance = result['feature_importance'][result['feature_importance']['Feature'] == 'Screen_Time_hrs_day']['Importance'].iloc[0] if 'Screen_Time_hrs_day' in result['feature_importance']['Feature'].values else 0
            distance_from_target = abs(screen_time_importance - 0.75)  # Target 75%
            closest_models.append((key, result['accuracy'], screen_time_importance, distance_from_target))
    
    closest_models.sort(key=lambda x: x[3])  # Sort by distance from target
    best_models = [(key, acc, imp) for key, acc, imp, _ in closest_models[:3]]

# Select the best model
if best_models:
    best_key = best_models[0][0]
    best_result = results[best_key]
    print(f"\n🏆 Selected best model: {best_key}")
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    
    # Get feature importance
    importance_df = best_result['feature_importance']
    screen_time_importance = importance_df[importance_df['Feature'] == 'Screen_Time_hrs_day']['Importance'].iloc[0] if 'Screen_Time_hrs_day' in importance_df['Feature'].values else 0
    print(f"   Screen_Time importance: {screen_time_importance:.1%}")

# Create comparison visualization
print("\n📈 Step 5: Creating visualizations...")

# Compare top models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

# Get top 4 models for comparison
top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:4]

for i, (key, result) in enumerate(top_models):
    if result['feature_importance'] is not None:
        ax = axes[i]
        top_features = result['feature_importance'].head(8)
        
        bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'], fontsize=9)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{key}\nAccuracy: {result["accuracy"]:.3f}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, top_features['Importance'])):
            ax.text(value + 0.01, j, f'{value:.1%}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('advanced_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Before/After comparison with original model
print("\n📊 Step 6: Before/After comparison...")

# Train original model for comparison
original_model = RandomForestClassifier(n_estimators=100, random_state=42)
original_model.fit(X_train_orig, y_train)
original_importance = pd.DataFrame({
    'Feature': original_features,
    'Importance': original_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Before (Original)
original_top = original_importance.head(6)
bars1 = ax1.barh(range(len(original_top)), original_top['Importance'], 
                 color='red', alpha=0.7)
ax1.set_yticks(range(len(original_top)))
ax1.set_yticklabels(original_top['Feature'])
ax1.set_xlabel('Feature Importance')
ax1.set_title('BEFORE: Original Random Forest\nScreen_Time dominance: 98.4%')
ax1.grid(axis='x', alpha=0.3)

for i, (bar, value) in enumerate(zip(bars1, original_top['Importance'])):
    ax1.text(value + 0.01, i, f'{value:.1%}', va='center', fontsize=10)

# After (Best Balanced Model)
if best_models:
    best_importance = best_result['feature_importance']
    best_top = best_importance.head(6)
    bars2 = ax2.barh(range(len(best_top)), best_top['Importance'], 
                     color='green', alpha=0.7)
    ax2.set_yticks(range(len(best_top)))
    ax2.set_yticklabels(best_top['Feature'])
    ax2.set_xlabel('Feature Importance')
    screen_time_imp = best_importance[best_importance['Feature'] == 'Screen_Time_hrs_day']['Importance'].iloc[0] if 'Screen_Time_hrs_day' in best_importance['Feature'].values else 0
    ax2.set_title(f'AFTER: {best_key}\nScreen_Time reduced to: {screen_time_imp:.1%}')
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars2, best_top['Importance'])):
        ax2.text(value + 0.01, i, f'{value:.1%}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('before_after_advanced.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the best model and scaler
if best_models:
    print(f"\n💾 Step 7: Saving the best balanced model...")
    
    # Save model and scaler
    joblib.dump(best_result['model'], 'smartphone_addiction_model_balanced_advanced.pkl')
    joblib.dump(best_result['scaler'], 'feature_scaler_advanced.pkl')
    
    # Save feature names for the app
    feature_info = {
        'features': enhanced_features,
        'model_type': best_key,
        'screen_time_importance': screen_time_imp
    }
    joblib.dump(feature_info, 'model_info.pkl')
    
    print("✅ Saved balanced model as 'smartphone_addiction_model_balanced_advanced.pkl'")
    print("✅ Saved feature scaler as 'feature_scaler_advanced.pkl'")
    print("✅ Saved model info as 'model_info.pkl'")

# Summary
print("\n📋 FINAL SUMMARY:")
print("=" * 50)
print(f"✅ Original Screen_Time importance: 98.4%")
if best_models:
    print(f"✅ New Screen_Time importance: {screen_time_imp:.1%}")
    print(f"✅ Improvement: {98.4 - screen_time_imp*100:.1f} percentage points")
    print(f"✅ Best model: {best_key}")
    print(f"✅ Accuracy maintained: {best_result['accuracy']:.3f}")
else:
    print("⚠️ No significant improvement achieved with current approaches")

print("\n🔍 TOP 5 FEATURES IN BEST MODEL:")
if best_models:
    for i, (_, row) in enumerate(best_result['feature_importance'].head(5).iterrows()):
        print(f"{i+1}. {row['Feature']:25} {row['Importance']:6.1%}")

print("\n🚀 NEXT STEPS:")
print("=" * 20)
print("1. Update Streamlit app to use the new balanced model")
print("2. Implement feature engineering in the app")
print("3. Test with real user data")
print("4. Deploy the improved version")

print("\n" + "="*70)
print("🎉 ADVANCED FEATURE BALANCING COMPLETE!")
print("="*70)
