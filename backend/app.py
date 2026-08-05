# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

print("="*50)
print("Loading models...")
print("="*50)

model_path = os.path.join(project_root, 'models', 'aggressive_best_model.pkl')
if not os.path.exists(model_path):
    model_path = os.path.join(project_root, 'models', 'best_model_Logistic_Regression.pkl')

preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print(f"✅ Model loaded")

print(f"Loading preprocessor from: {preprocessor_path}")
preprocessor = joblib.load(preprocessor_path)
print("✅ Preprocessor loaded")

def preprocess_customer_data(data):
    df = pd.DataFrame([data])
    
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
    
    df = df.drop('TotalCharges', axis=1)
    
    df['Tenure_Contract_Interaction'] = df['tenure'] * (df['Contract'] == 'Month-to-month').astype(int)
    df['Charges_Senior_Interaction'] = df['MonthlyCharges'] * df['SeniorCitizen']
    df['Tenure_Charges_Ratio'] = df['tenure'] / (df['MonthlyCharges'] + 0.01)
    df['Is_Fiber_Optics'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Is_Electronic_Check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    return preprocessor.transform(df)

def get_recommendations(data):
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
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X_processed = preprocess_customer_data(data)
        prob = model.predict_proba(X_processed)[0][1]
        prob_percent = round(prob * 100, 2)
        
        if prob >= 0.7:
            risk = "High"
        elif prob >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"
        
        recommendations = get_recommendations(data)
        
        return jsonify({
            'probability': prob_percent,
            'risk_level': risk,
            'recommendations': recommendations
        })
    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        print("=== BATCH PREDICTION STARTED ===")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Processing file: {file.filename}")
        
        # Read CSV
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except:
            df = pd.read_csv(file, encoding='latin1')
        
        print(f"Loaded {len(df)} rows from CSV")
        
        # Create case-insensitive column mapping
        col_map = {col.lower(): col for col in df.columns}
        
        # Helper to get column values with proper type handling
        def get_col_str(key, default):
            if key.lower() in col_map:
                col = col_map[key.lower()]
                if col in df.columns:
                    # Strip whitespace and replace empty strings with default
                    return df[col].astype(str).str.strip().replace('', default).fillna(default)
            return pd.Series([default] * len(df), index=df.index)
        
        def get_col_num(key, default):
            if key.lower() in col_map:
                col = col_map[key.lower()]
                if col in df.columns:
                    # Convert to numeric, coerce errors to NaN, then fill with default
                    return pd.to_numeric(df[col], errors='coerce').fillna(default)
            return pd.Series([default] * len(df), index=df.index)
        
        # Extract columns
        gender = get_col_str('gender', 'Male')
        senior = get_col_num('SeniorCitizen', 0).astype(int)
        partner = get_col_str('Partner', 'No')
        dependents = get_col_str('Dependents', 'No')
        tenure = get_col_num('tenure', 0)
        phone = get_col_str('PhoneService', 'No')
        multiple_lines = get_col_str('MultipleLines', 'No')
        internet = get_col_str('InternetService', 'No')
        online_sec = get_col_str('OnlineSecurity', 'No')
        online_back = get_col_str('OnlineBackup', 'No')
        device_prot = get_col_str('DeviceProtection', 'No')
        tech_support = get_col_str('TechSupport', 'No')
        streaming_tv = get_col_str('StreamingTV', 'No')
        streaming_movies = get_col_str('StreamingMovies', 'No')
        contract = get_col_str('Contract', 'Month-to-month')
        paperless = get_col_str('PaperlessBilling', 'No')
        payment = get_col_str('PaymentMethod', 'Electronic check')
        monthly = get_col_num('MonthlyCharges', 70)
        total = get_col_num('TotalCharges', 0)
        
        # Build DataFrame
        df_clean = pd.DataFrame({
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_back,
            'DeviceProtection': device_prot,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        })
        
        # Feature engineering (vectorized)
        df_fe = df_clean.copy()
        df_fe['Avg_Monthly_Spend'] = df_fe['TotalCharges'] / (df_fe['tenure'] + 0.01)
        df_fe['Tenure_Group'] = pd.cut(
            df_fe['tenure'],
            bins=[0, 6, 24, 100],
            labels=['New', 'Regular', 'Veteran']
        )
        
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        df_fe['Service_Count'] = df_fe[service_cols].apply(lambda row: sum(row != 'No'), axis=1)
        df_fe['High_Risk_Flag'] = ((df_fe['Contract'] == 'Month-to-month') &
                                   (df_fe['MonthlyCharges'] > 70)).astype(int)
        
        df_fe = df_fe.drop('TotalCharges', axis=1)
        
        # Additional features
        df_fe['Tenure_Contract_Interaction'] = df_fe['tenure'] * (df_fe['Contract'] == 'Month-to-month').astype(int)
        df_fe['Charges_Senior_Interaction'] = df_fe['MonthlyCharges'] * df_fe['SeniorCitizen']
        df_fe['Tenure_Charges_Ratio'] = df_fe['tenure'] / (df_fe['MonthlyCharges'] + 0.01)
        df_fe['Is_Fiber_Optics'] = (df_fe['InternetService'] == 'Fiber optic').astype(int)
        df_fe['Is_Electronic_Check'] = (df_fe['PaymentMethod'] == 'Electronic check').astype(int)
        
        # Preprocess all at once
        X_processed = preprocessor.transform(df_fe)
        
        # Predict
        probs = model.predict_proba(X_processed)[:, 1]
        prob_percents = (probs * 100).round(2)
        
        risk_levels = np.where(probs >= 0.7, 'High',
                              np.where(probs >= 0.4, 'Medium', 'Low'))
        
        # Build results
        results = []
        for idx, row in df_clean.iterrows():
            data = row.to_dict()
            recs = get_recommendations(data)
            results.append({
                'row': idx + 1,
                'probability': float(prob_percents[idx]),
                'risk_level': risk_levels[idx],
                'recommendations': recs,
                'customer_data': data
            })
        
        high_risk = sum(risk_levels == 'High')
        medium_risk = sum(risk_levels == 'Medium')
        low_risk = sum(risk_levels == 'Low')
        
        print(f"Results: High={high_risk}, Medium={medium_risk}, Low={low_risk}")
        
        return jsonify({
            'total': len(results),
            'high_risk': int(high_risk),
            'medium_risk': int(medium_risk),
            'low_risk': int(low_risk),
            'results': results
        })
        
    except Exception as e:
        print(f"Error in /predict_batch: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400 
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Flask server starting...")
    print("="*50)
    print("API endpoints:")
    print("  POST /predict        - Predict single customer")
    print("  POST /predict_batch  - Predict batch customers from CSV")
    print("  GET  /health         - Health check")
    print("\n" + "="*50)
    app.run(host='0.0.0.0', port=5000, debug=True)