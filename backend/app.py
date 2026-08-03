# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Get project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

print("="*50)
print("Loading models...")
print("="*50)

# Load model and preprocessor
model_path = os.path.join(project_root, 'models', 'aggressive_best_model.pkl')
if not os.path.exists(model_path):
    model_path = os.path.join(project_root, 'models', 'best_model_Logistic_Regression.pkl')

preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print(f"✅ Model loaded (expects {model.n_features_in_} features)")

print(f"Loading preprocessor from: {preprocessor_path}")
preprocessor = joblib.load(preprocessor_path)
print("✅ Preprocessor loaded")

def preprocess_customer_data(data):
    """Convert JSON input to DataFrame with ALL feature engineering"""
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # ============================================
    # ORIGINAL FEATURE ENGINEERING (from Section 3)
    # ============================================
    df['Avg_Monthly_Spend'] = df['TotalCharges'] / (df['tenure'] + 0.01)
    df['Tenure_Group'] = pd.cut(
        df['tenure'],
        bins=[0, 6, 24, 100],
        labels=['New', 'Regular', 'Veteran']
    )
    
    service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = df[service_cols].apply(lambda row: sum(row != 'No'), axis=1)
    df['High_Risk_Flag'] = ((df['Contract'] == 'Month-to-month') &
                            (df['MonthlyCharges'] > 70)).astype(int)
    
    # Drop TotalCharges (since we have Avg_Monthly_Spend)
    df = df.drop('TotalCharges', axis=1)
    
    # ============================================
    # FEATURE ENGINEERING 2.0 (from Section 4 - Aggressive Tuning)
    # ============================================
    df['Tenure_Contract_Interaction'] = df['tenure'] * (df['Contract'] == 'Month-to-month').astype(int)
    df['Charges_Senior_Interaction'] = df['MonthlyCharges'] * df['SeniorCitizen']
    df['Tenure_Charges_Ratio'] = df['tenure'] / (df['MonthlyCharges'] + 0.01)
    df['Is_Fiber_Optics'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Is_Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # ============================================
    # Apply Preprocessor
    # ============================================
    X_processed = preprocessor.transform(df)
    
    # Debug: print shapes
    print(f"Processed shape: {X_processed.shape}")
    print(f"Expected features: {model.n_features_in_}")
    
    return X_processed

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data with keys: {list(data.keys())}")
        
        # Preprocess
        X_processed = preprocess_customer_data(data)
        
        # Predict probability
        prob = model.predict_proba(X_processed)[0][1]
        prob_percent = round(prob * 100, 2)
        
        # Risk level
        if prob >= 0.7:
            risk = "High"
        elif prob >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"
        
        # Generate recommendations
        recommendations = []
        if data.get('Contract') == 'Month-to-month':
            recommendations.append("Offer 12-month contract with 10% discount")
        if data.get('OnlineSecurity') == 'No':
            recommendations.append("Provide free Online Security add-on for 3 months")
        if data.get('TechSupport') == 'No':
            recommendations.append("Offer free Tech Support trial")
        if data.get('tenure', 0) < 6:
            recommendations.append("Send welcome series with engagement tips")
        if data.get('MonthlyCharges', 0) > 70:
            recommendations.append("Suggest plan optimization to reduce bill")
        if not recommendations:
            recommendations.append("Send general satisfaction survey")
        
        return jsonify({
            'probability': prob_percent,
            'risk_level': risk,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask server starting...")
    print("="*50)
    print(f"Model type: {type(model).__name__}")
    print(f"Features expected: {model.n_features_in_}")
    print("\nAPI endpoints:")
    print("  POST /predict  - Predict churn risk")
    print("  GET  /health   - Health check")
    print("\n" + "="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)