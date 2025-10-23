"""
Streamlit Cloud Deployment Helper Script
=======================================

This script helps you deploy the smartphone addiction predictor to Streamlit Cloud.
It provides instructions and checks for deployment readiness.

Author: AI Assistant
Date: 2025
"""

import os
import sys

def check_deployment_readiness():
    """Check if all required files are present for deployment"""
    print("🔍 CHECKING DEPLOYMENT READINESS")
    print("=" * 50)
    
    required_files = [
        'app_redistributed.py',
        'smartphone_addiction_model_redistributed.pkl',
        'feature_scaler_redistributed.pkl',
        'model_info_redistributed.pkl',
        'phone_usage_india.csv',
        'requirements.txt'
    ]
    
    missing_files = []
    present_files = []
    
    for file in required_files:
        if os.path.exists(file):
            present_files.append(file)
            print(f"✅ {file}")
        else:
            missing_files.append(file)
            print(f"❌ {file}")
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Present: {len(present_files)}/{len(required_files)} files")
    print(f"❌ Missing: {len(missing_files)} files")
    
    if missing_files:
        print(f"\n⚠️ MISSING FILES:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print(f"\n🎉 ALL FILES PRESENT - READY FOR DEPLOYMENT!")
        return True

def show_deployment_instructions():
    """Show step-by-step deployment instructions"""
    print("\n🚀 STREAMLIT CLOUD DEPLOYMENT INSTRUCTIONS")
    print("=" * 50)
    
    print("""
1. 🌐 Go to: https://share.streamlit.io
2. 🔐 Sign in with your GitHub account
3. ➕ Click "New app"
4. 📝 Fill in the details:
   - Repository: bchethan/smartphone_addiction_prediction
   - Branch: main
   - Main file path: app_redistributed.py
5. 🚀 Click "Deploy"
6. ⏳ Wait for deployment (2-3 minutes)
7. 🎉 Access your app at the provided URL

📱 RECOMMENDED VERSION: app_redistributed.py
   - 56.5% Screen_Time importance (vs 98.4% original)
   - 22 redistributed features
   - Best balance and interpretability
   - Fixed text visibility issues
""")

def show_alternative_deployments():
    """Show alternative deployment options"""
    print("\n🔄 ALTERNATIVE DEPLOYMENT OPTIONS")
    print("=" * 40)
    
    print("""
📱 VERSION 1: Original Model
   - Main file: app.py
   - Screen_Time importance: 98.4%
   - Features: 11
   - URL: https://share.streamlit.io/bchethan/smartphone_addiction_prediction/main/app.py

⚖️ VERSION 2: Balanced Model  
   - Main file: app_balanced.py
   - Screen_Time importance: 69.7%
   - Features: 21
   - URL: https://share.streamlit.io/bchethan/smartphone_addiction_prediction/main/app_balanced.py

🎯 VERSION 3: Redistributed Model (RECOMMENDED)
   - Main file: app_redistributed.py
   - Screen_Time importance: 56.5%
   - Features: 22
   - URL: https://share.streamlit.io/bchethan/smartphone_addiction_prediction/main/app_redistributed.py
""")

def main():
    """Main deployment helper function"""
    print("📱 SMARTPHONE ADDICTION PREDICTOR - DEPLOYMENT HELPER")
    print("=" * 60)
    
    # Check deployment readiness
    is_ready = check_deployment_readiness()
    
    if is_ready:
        show_deployment_instructions()
        show_alternative_deployments()
        
        print("\n🎯 QUICK DEPLOYMENT COMMAND:")
        print("=" * 30)
        print("1. Go to: https://share.streamlit.io")
        print("2. New app → Repository: bchethan/smartphone_addiction_prediction")
        print("3. Branch: main → Main file: app_redistributed.py")
        print("4. Deploy!")
        
        print("\n✅ READY TO DEPLOY!")
    else:
        print("\n❌ NOT READY - Please ensure all required files are present")
        print("Run the training scripts to generate missing model files:")
        print("   python redistributed_feature_balancing.py")

if __name__ == "__main__":
    main()
