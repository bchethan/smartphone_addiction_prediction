import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Smartphone Addiction Predictor",
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
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📱 Smartphone Addiction Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze your smartphone usage patterns and get insights about potential addiction risk</p>', unsafe_allow_html=True)

# Function to load the model
@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = joblib.load('smartphone_addiction_model.pkl')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'smartphone_addiction_model.pkl' not found!")
        st.info("Please ensure the model file is in the same directory as this app.")
        return None

# Function to encode categorical variables
def encode_features(gender, location):
    """Encode gender and location as per training specifications"""
    # Gender encoding: Male = 1, Female = 0
    gender_encoded = 1 if gender == "Male" else 0
    
    # Location encoding: Urban = 2, Semi-Urban = 1, Rural = 0
    location_mapping = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}
    location_encoded = location_mapping.get(location, 0)
    
    return gender_encoded, location_encoded

# Function to create feature importance visualization
def plot_feature_importance(model, feature_names):
    """Plot top 3 feature importances if model supports it"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(3)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Top 3 Features Contributing to Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        return fig
    return None

# Load the model
model = load_model()

if model is not None:
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
        
        # Create feature array in the exact order expected by the model
        features = [
            age, gender_encoded, location_encoded, screen_time, data_usage,
            call_duration, apps_installed, social_media_time, streaming_time,
            gaming_time, monthly_recharge
        ]
        
        # Feature names for display
        feature_names = [
            'Age', 'Gender_encoded', 'Location_encoded', 'Screen_Time_hrs_day',
            'Data_Usage_GB_month', 'Calls_Duration_mins_day', 'Number_of_Apps_Installed',
            'Social_Media_Time_hrs_day', 'Streaming_Time_hrs_day', 'Gaming_Time_hrs_day',
            'Monthly_Recharge_Cost_INR'
        ]
        
        # Make prediction
        try:
            # Reshape features for prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Get prediction and probability
            prediction = model.predict(features_array)[0]
            probability = model.predict_proba(features_array)[0]
            
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
            
            # Feature importance visualization
            st.markdown("## 📊 Feature Analysis")
            importance_fig = plot_feature_importance(model, feature_names)
            if importance_fig:
                st.pyplot(importance_fig)
                st.caption("Top 3 features that most influence the prediction")
            else:
                st.info("Feature importance visualization not available for this model type.")
            
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
            st.info("Please check if the model file is compatible with the expected feature format.")
    
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
    st.error("❌ Unable to load the prediction model. Please ensure 'smartphone_addiction_model.pkl' is available.")
    st.info("""
    To create a model file, you can:
    1. Train a model using the provided dataset
    2. Save it using: `joblib.dump(model, 'smartphone_addiction_model.pkl')`
    3. Place the file in the same directory as this app
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
Made with ❤️ using Streamlit | Smartphone Addiction Predictor
</div>
""", unsafe_allow_html=True)
