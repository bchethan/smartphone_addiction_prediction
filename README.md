# 📱 Smartphone Addiction Predictor

A comprehensive Streamlit web application that predicts smartphone addiction risk based on user behavioral and demographic data.

## 🚀 Features

- **Interactive Input Forms**: Easy-to-use sliders and dropdowns for all user data
- **Real-time Prediction**: Instant addiction risk assessment with confidence scores
- **Feature Analysis**: Visualization of top contributing factors
- **Digital Wellness Tips**: Personalized recommendations for users at risk
- **Beautiful UI**: Modern, responsive design with intuitive navigation

## 📊 Input Parameters

The app collects the following user data:

### Personal Information
- **Age**: 15-80 years (slider)
- **Gender**: Male/Female (dropdown)
- **Location**: Urban/Semi-Urban/Rural (dropdown)

### Usage Patterns
- **Screen Time**: 0-24 hours/day (slider)
- **Social Media Time**: 0-12 hours/day (slider)
- **Gaming Time**: 0-8 hours/day (slider)
- **Streaming Time**: 0-8 hours/day (slider)
- **Call Duration**: 0-300 minutes/day (slider)
- **Data Usage**: 0-100 GB/month (slider)
- **Apps Installed**: 10-200 apps (slider)
- **Monthly Recharge Cost**: ₹100-₹2000 (slider)

## 🛠️ Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

First, train the machine learning model using the provided dataset:

```bash
python train_model.py
```

This will:
- Load the `phone_usage_india.csv` dataset
- Train a Random Forest classifier
- Save the model as `smartphone_addiction_model.pkl`
- Display model performance metrics

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 🎯 How to Use

1. **Fill the Form**: Use the sidebar to input your smartphone usage data
2. **Submit**: Click "Predict Addiction Risk" to get your analysis
3. **Review Results**: Check your risk level, confidence score, and feature analysis
4. **Get Tips**: If at risk, follow the digital wellness recommendations

## 📈 Model Details

### Features Used
The model expects features in this exact order:
```
['Age', 'Gender_encoded', 'Location_encoded', 'Screen_Time_hrs_day', 
 'Data_Usage_GB_month', 'Calls_Duration_mins_day', 'Number_of_Apps_Installed', 
 'Social_Media_Time_hrs_day', 'Streaming_Time_hrs_day', 
 'Gaming_Time_hrs_day', 'Monthly_Recharge_Cost_INR']
```

### Encoding Scheme
- **Gender**: Male = 1, Female = 0
- **Location**: Urban = 2, Semi-Urban = 1, Rural = 0

### Addiction Criteria
- Users with screen time > 8 hours/day are classified as addicted
- The model provides both binary classification and probability scores

## 📊 Output Interpretation

### Risk Levels
- **🟢 Low Risk**: < 40% probability
- **🟡 Medium Risk**: 40-60% probability  
- **🟠 High Risk**: 60-80% probability
- **🔴 Very High Risk**: > 80% probability

### Results Display
- **Binary Classification**: Addicted/Not Addicted
- **Confidence Score**: Probability percentage
- **Feature Importance**: Top 3 contributing factors
- **Input Summary**: Review of entered data

## 💡 Digital Wellness Tips

For users identified as at-risk, the app provides actionable tips:

- Set screen time limits using built-in phone features
- Create phone-free zones (bedroom, dining areas)
- Use grayscale mode to reduce visual appeal
- Practice regular digital detox periods
- Find offline hobbies and activities
- Organize and declutter your home screen
- Limit non-essential notifications
- Track and monitor usage patterns

## ⚠️ Disclaimer

This tool is for **educational purposes only** and should not replace professional medical advice. If you're concerned about your smartphone usage, consider consulting with a mental health professional.

## 🛠️ Technical Details

### Technologies Used
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Features**: 11 behavioral and demographic variables
- **Target**: Binary classification (addicted/not addicted)
- **Evaluation**: Accuracy, precision, recall, F1-score

## 📁 Project Structure

```
smartphone/
├── app.py                    # Main Streamlit application
├── train_model.py           # Model training script
├── main.py                  # Original data analysis script
├── phone_usage_india.csv    # Dataset
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── smartphone_addiction_model.pkl  # Trained model (generated)
```

## 🔧 Troubleshooting

### Common Issues

1. **Model file not found**
   - Run `python train_model.py` first
   - Ensure `smartphone_addiction_model.pkl` exists in the project directory

2. **Dependencies missing**
   - Install requirements: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Port already in use**
   - Use different port: `streamlit run app.py --server.port 8502`
   - Or kill existing process on port 8501

## 🤝 Contributing

Feel free to contribute by:
- Improving the UI/UX
- Adding new features
- Enhancing the model performance
- Adding more wellness tips
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ using Streamlit | Smartphone Addiction Predictor**
