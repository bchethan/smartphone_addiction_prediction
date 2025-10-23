import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Smartphone Addiction Predictor - Balanced Model",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .tips-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .improvement-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📱 Smartphone Addiction Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML model with balanced feature importance - Screen Time dominance reduced from 98.4% to 69.7%</p>', unsafe_allow_html=True)

# Show improvement banner
st.markdown("""
<div class="improvement-box" style="color: #155724;">
<h4 style="color: #155724;">🎯 Model Improvement:</h4>
<p style="color: #155724;"><strong>✅ Balanced Feature Importance:</strong> Reduced Screen_Time dominance from 98.4% to 69.7%</p>
<p style="color: #155724;"><strong>✅ Enhanced Features:</strong> Added 10 new engineered features for better predictions</p>
<p style="color: #155724;"><strong>✅ Advanced Algorithm:</strong> Using Gradient Boosting with StandardScaler for optimal performance</p>
</div>
""", unsafe_allow_html=True)

# Function to load the balanced model and scaler
@st.cache_resource
def load_balanced_model():
    """Load the balanced model, scaler, and feature info"""
    try:
        model = joblib.load('smartphone_addiction_model_balanced_advanced.pkl')
        scaler = joblib.load('feature_scaler_advanced.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("❌ Balanced model files not found!")
        st.info("Please run 'python advanced_feature_balancing.py' first to generate the balanced model.")
        return None, None, None

# Function to encode categorical variables
def encode_features(gender, location):
    """Encode gender and location as per training specifications"""
    # Gender encoding: Male = 1, Female = 0
    gender_encoded = 1 if gender == "Male" else 0
    
    # Location encoding: Urban = 2, Semi-Urban = 1, Rural = 0
    location_mapping = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
    location_encoded = location_mapping.get(location, 0)
    
    return gender_encoded, location_encoded

# Function to create enhanced features
def create_enhanced_features(user_inputs):
    """Create enhanced features from user inputs"""
    age, gender_encoded, location_encoded, screen_time, data_usage, call_duration, apps_installed, social_media_time, streaming_time, gaming_time, monthly_recharge = user_inputs
    
    # Create engineered features (same as in training)
    total_entertainment_time = social_media_time + streaming_time + gaming_time
    screen_time_ratio = screen_time / 24.0
    data_usage_per_hour = data_usage / (screen_time * 30 + 1)  # Avoid division by zero
    app_density = apps_installed / 100.0
    cost_per_hour = monthly_recharge / (screen_time * 30 + 1)
    communication_intensity = call_duration / 60.0  # Convert to hours
    
    # Age group encoding
    if age <= 25:
        age_group = 0
    elif age <= 35:
        age_group = 1
    elif age <= 50:
        age_group = 2
    else:
        age_group = 3
    
    # Interaction features
    screen_time_x_apps = screen_time * app_density
    entertainment_x_data = total_entertainment_time * data_usage
    age_x_screen_time = age * screen_time
    
    # Return all features in the correct order
    enhanced_features = [
        age, gender_encoded, location_encoded,
        screen_time, screen_time_ratio,
        data_usage, data_usage_per_hour,
        call_duration, communication_intensity,
        apps_installed, app_density,
        social_media_time, streaming_time, gaming_time,
        total_entertainment_time,
        monthly_recharge, cost_per_hour,
        age_group,
        screen_time_x_apps, entertainment_x_data, age_x_screen_time
    ]
    
    return enhanced_features

# Function to create feature importance visualization
def plot_feature_importance(model, feature_names, top_n=8):
    """Plot top N feature importances"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
                       color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df))))
        
        ax.set_yticks(range(len(feature_importance_df)))
        ax.set_yticklabels(feature_importance_df['Feature'], fontsize=11)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top Features Contributing to Prediction (Balanced Model)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, feature_importance_df['Importance'])):
            ax.text(value + 0.01, i, f'{value:.1%}', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        return fig
    return None

# Load the balanced model
model, scaler, model_info = load_balanced_model()

if model is not None and scaler is not None and model_info is not None:
    # Create sidebar for user inputs
    st.sidebar.markdown("## 📊 Enter Your Data")
    
    # User input form
    with st.sidebar.form("prediction_form"):
        st.markdown("### Personal Information")
        age = st.slider("Age", min_value=15, max_value=80, value=25, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location", ["Urban", "Semi-Urban", "Rural"])
        
        st.markdown("### Usage Patterns")
        screen_time = st.slider("Screen Time (hours/day)", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
        social_media_time = st.slider("Social Media Time (hours/day)", min_value=0.0, max_value=12.0, value=2.0, step=0.5)
        gaming_time = st.slider("Gaming Time (hours/day)", min_value=0.0, max_value=8.0, value=1.0, step=0.5)
        streaming_time = st.slider("Streaming Time (hours/day)", min_value=0.0, max_value=8.0, value=2.0, step=0.5)
        call_duration = st.slider("Call Duration (minutes/day)", min_value=0, max_value=300, value=60, step=10)
        data_usage = st.slider("Data Usage (GB/month)", min_value=0, max_value=100, value=20, step=1)
        apps_installed = st.slider("Number of Apps Installed", min_value=10, max_value=200, value=50, step=5)
        monthly_recharge = st.slider("Monthly Recharge Cost (INR)", min_value=100, max_value=2000, value=500, step=50)
        
        submit_button = st.form_submit_button("🔍 Predict Addiction Risk")
    
    # Main content area
    if submit_button:
        st.markdown("## 🎯 Prediction Results")
        
        # Encode categorical variables
        gender_encoded, location_encoded = encode_features(gender, location)
        
        # Create user inputs array
        user_inputs = [
            age, gender_encoded, location_encoded, screen_time, data_usage,
            call_duration, apps_installed, social_media_time, streaming_time,
            gaming_time, monthly_recharge
        ]
        
        # Create enhanced features
        enhanced_features = create_enhanced_features(user_inputs)
        
        # Make prediction
        try:
            # Reshape features for prediction
            features_array = np.array(enhanced_features).reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features_array)
            
            # Get prediction and probability
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Prediction")
                if prediction == 1:
                    st.error("🚨 **HIGH RISK: Likely Addicted**")
                    st.markdown(f"**Confidence:** {probability[1]:.1%}")
                else:
                    st.success("✅ **LOW RISK: Not Addicted**")
                    st.markdown(f"**Confidence:** {probability[0]:.1%}")
            
            with col2:
                st.markdown("### 📈 Risk Breakdown")
                risk_score = probability[1] if prediction == 1 else probability[0]
                st.metric("Risk Score", f"{risk_score:.1%}")
                
                # Risk level indicator
                if risk_score > 0.8:
                    risk_level = "Very High"
                    color = "🔴"
                elif risk_score > 0.6:
                    risk_level = "High"
                    color = "🟠"
                elif risk_score > 0.4:
                    risk_level = "Medium"
                    color = "🟡"
                else:
                    risk_level = "Low"
                    color = "🟢"
                
                st.markdown(f"**Risk Level:** {color} {risk_level}")
                
                # Show model info
                st.markdown("### 🤖 Model Info")
                st.markdown(f"**Algorithm:** {model_info['model_type']}")
                st.markdown(f"**Features:** {len(model_info['features'])} enhanced features")
                st.markdown(f"**Screen_Time Importance:** {model_info['screen_time_importance']:.1%}")
            
            # Feature importance visualization
            st.markdown("## 📊 Feature Analysis (Balanced Model)")
            importance_fig = plot_feature_importance(model, model_info['features'])
            if importance_fig:
                st.pyplot(importance_fig)
                st.caption("Top features that most influence the prediction (with balanced importance)")
            else:
                st.info("Feature importance visualization not available for this model type.")
            
            # Show feature engineering details
            st.markdown("## 🔧 Enhanced Features Used")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Features:**")
                st.write("• Age, Gender, Location")
                st.write("• Screen Time, Data Usage, Call Duration")
                st.write("• Apps Installed, Social Media, Gaming, Streaming")
                st.write("• Monthly Recharge Cost")
            
            with col2:
                st.markdown("**Engineered Features:**")
                st.write("• Screen Time Ratio (screen_time/24)")
                st.write("• Total Entertainment Time")
                st.write("• Data Usage per Hour")
                st.write("• App Density, Cost per Hour")
                st.write("• Age Group, Communication Intensity")
                st.write("• Interaction Features (Screen×Apps, etc.)")
            
            # Digital wellness tips for addicted users
            if prediction == 1:
                st.markdown("## 💡 Digital Wellness Tips")
                st.markdown("""
                <div class="tips-box" style="color: #1a365d;">
                <h4 style="color: #1a365d;">🚨 Since you're at risk of smartphone addiction, here are some tips:</h4>
                <ul style="color: #1a365d;">
                <li><strong>Set Screen Time Limits:</strong> Use your phone's built-in screen time features to set daily limits</li>
                <li><strong>Create Phone-Free Zones:</strong> Keep your phone out of the bedroom and dining areas</li>
                <li><strong>Use Grayscale Mode:</strong> Switch to black and white to make your phone less appealing</li>
                <li><strong>Practice Digital Detox:</strong> Take regular breaks from your phone, especially before bed</li>
                <li><strong>Find Offline Hobbies:</strong> Replace screen time with physical activities or reading</li>
                <li><strong>Use Apps Mindfully:</strong> Delete apps you don't use and organize your home screen</li>
                <li><strong>Set Notification Limits:</strong> Turn off non-essential notifications</li>
                <li><strong>Track Your Usage:</strong> Use apps to monitor and reduce your screen time</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Display input summary
            st.markdown("## 📋 Your Input Summary")
            summary_data = {
                "Personal Info": [f"Age: {age}", f"Gender: {gender}", f"Location: {location}"],
                "Usage Patterns": [
                    f"Screen Time: {screen_time} hrs/day",
                    f"Social Media: {social_media_time} hrs/day",
                    f"Gaming: {gaming_time} hrs/day",
                    f"Streaming: {streaming_time} hrs/day",
                    f"Call Duration: {call_duration} mins/day",
                    f"Data Usage: {data_usage} GB/month",
                    f"Apps Installed: {apps_installed}",
                    f"Monthly Recharge: ₹{monthly_recharge}"
                ]
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Personal Information:**")
                for info in summary_data["Personal Info"]:
                    st.write(f"• {info}")
            
            with col2:
                st.markdown("**Usage Patterns:**")
                for pattern in summary_data["Usage Patterns"]:
                    st.write(f"• {pattern}")
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check if the model files are compatible with the expected feature format.")
    
    else:
        # Show welcome message and instructions
        st.markdown("## 🎯 How to Use This App")
        st.markdown("""
        1. **Fill out the form** in the sidebar with your smartphone usage data
        2. **Click 'Predict Addiction Risk'** to get your analysis
        3. **Review the results** including risk level and confidence score
        4. **Check the feature analysis** to understand what influences the prediction
        5. **Follow wellness tips** if you're at risk of addiction
        """)
        
        # Show model improvements
        st.markdown("## 🚀 Model Improvements")
        st.markdown("""
        This version uses an **advanced balanced model** with the following improvements:
        
        ### ✅ **Feature Engineering**
        - **21 enhanced features** (vs 11 original)
        - **Screen Time Ratio**: Normalized screen time (screen_time/24)
        - **Total Entertainment Time**: Combined social media, gaming, streaming
        - **Data Usage per Hour**: Efficiency metric
        - **Interaction Features**: Screen×Apps, Entertainment×Data, Age×Screen
        
        ### ✅ **Algorithm Upgrade**
        - **Gradient Boosting** (vs Random Forest)
        - **StandardScaler** for optimal feature scaling
        - **Balanced Feature Importance**: Screen_Time reduced from 98.4% to 69.7%
        
        ### ✅ **Better Interpretability**
        - More meaningful feature contributions
        - Enhanced prediction accuracy
        - Improved model transparency
        """)
        
        # Add some statistics or information
        st.markdown("## 📊 About Smartphone Addiction")
        st.markdown("""
        Smartphone addiction is characterized by excessive use that interferes with daily life. 
        Common indicators include:
        - Spending more than 8 hours per day on your phone
        - Feeling anxious when separated from your device
        - Neglecting real-world activities and relationships
        - Using phone during inappropriate times (meetings, meals, etc.)
        """)
        
        # Add a disclaimer
        st.markdown("---")
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; color: #856404;'>
        <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. 
        If you're concerned about your smartphone usage, consider consulting with a mental health professional.
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load the balanced prediction model. Please ensure the model files are available.")
    st.info("""
    To create the balanced model files, run:
    1. `python advanced_feature_balancing.py` - Creates the balanced model
    2. Ensure these files exist:
       - `smartphone_addiction_model_balanced_advanced.pkl`
       - `feature_scaler_advanced.pkl`
       - `model_info.pkl`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
Made with ❤️ using Streamlit | Smartphone Addiction Predictor - Balanced Model v2.0
</div>
""", unsafe_allow_html=True)
