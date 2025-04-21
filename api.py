import joblib
import warnings
import logging
import pandas as pd
from flask import Flask, request, jsonify

# Suppress warnings and logs
warnings.filterwarnings("ignore")
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Load model and scaler (assumed in same directory for Render)
model_path = 'best_model.pkl'
scaler_path = 'scaler.pkl'

loaded_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Selected features used during training
selected_features = ['INTUBED', 'AGE', 'PATIENT_TYPE', 'ICU', 'PNEUMONIA', 'DIABETES', 'HIPERTENSION']

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "COVID Prediction API is running on Render!"})

# 📍 Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_covid():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        input_df = input_df[selected_features]  # Ensure correct order

        # Apply the same scaler
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = loaded_model.predict(input_scaled)[0]
        proba = loaded_model.predict_proba(input_scaled)[0][1]

        return jsonify({
            "prediction": "Positive" if int(prediction) == 1 else "Negative",
            "probability": round(float(proba), 2),
            "status": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
