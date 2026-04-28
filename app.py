
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import json
import xgboost as xgb

app = Flask(__name__)

# Load artifacts
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# FIX: Robust CPU forcing for XGBoost 2.x and 3.x
# 'gpu_id' is removed; we must use 'device' and ensure the predictor is set to cpu
try:
    model.set_params(device='cpu', predictor='cpu_predictor')
except:
    # Fallback for older versions if needed
    if hasattr(model, 'set_params'):
        model.set_params(gpu_id=-1)

with open('feature_medians.pkl', 'rb') as f:
    feature_medians = pickle.load(f)

with open('model_config.json', 'r') as f:
    config = json.load(f)

API_KEY = os.getenv("API_KEY", "my-secret-key-123")

def get_severity(score):
    if score >= 75: return 'HIGH'
    if score >= 50: return 'MEDIUM'
    if score >= 25: return 'LOW'
    return 'NORMAL'

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        inp = data.get("invoice_features", data)

        features = config['features']
        x_vec = []
        for f in features:
            val = inp.get(f, feature_medians.get(f, 0))
            x_vec.append(float(val))

        # Force CPU at inference time via DMatrix if needed, 
        # but calling model.predict directly is usually fine once params are set.
        prediction = float(model.predict(np.array([x_vec]))[0])
        actual = float(inp.get('debit', 0))
        residual = actual - prediction

        score = round(min(abs(residual) / config['ml_threshold_gbp'] * 100, 100.0), 2)

        return jsonify({
            "is_anomaly": score >= 25.0,
            "anomaly_score": score,
            "anomaly_severity": get_severity(score),
            "recommended_action": "ESCALATE" if score >= 75 else "APPROVE",
            "actual_debit": round(actual, 2),
            "predicted_debit": round(prediction, 2),
            "deviation_gbp": round(residual, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Billing Anomaly API Active (CPU Mode)"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
