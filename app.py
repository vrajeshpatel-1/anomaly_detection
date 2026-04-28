
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import json
import shap
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# ᐅ LOAD MODEL ARTIFACTS
# -----------------------------
try:
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('feature_medians.pkl', 'rb') as f:
        feature_medians = pickle.load(f)

    with open('model_config.json', 'r') as f:
        config = json.load(f)

    explainer = shap.TreeExplainer(model)
    ML_FEATURES = config['features']
    ML_THRESHOLD = config['ml_threshold_gbp']
    RESIDUAL_STD = config['residual_std_gbp']
    VAT_RATE = 0.20
except Exception as e:
    print(f"Model loading failed: {e}")
    ML_FEATURES = []

# -----------------------------
# ፠ API KEY (env-based)
# -----------------------------
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
        actual_debit = float(inp.get('debit', 0.0))

        x_vec = []
        for f in ML_FEATURES:
            val = inp.get(f, feature_medians.get(f, 0))
            x_vec.append(float(val))

        x_vec = np.array([x_vec])
        predicted_debit = float(model.predict(x_vec)[0])

        residual_signed = actual_debit - predicted_debit
        abs_residual = abs(residual_signed)
        anomaly_score = round(min(abs_residual / ML_THRESHOLD * 100, 100.0), 2)

        result = {
            "is_anomaly": anomaly_score >= 25.0,
            "anomaly_score": anomaly_score,
            "anomaly_severity": get_severity(anomaly_score),
            "recommended_action": "ESCALATE" if anomaly_score >= 75 else "APPROVE",
            "actual_debit": round(actual_debit, 2),
            "predicted_debit": round(predicted_debit, 2),
            "deviation_gbp": round(residual_signed, 2)
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
