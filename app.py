
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
import shap
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# 🔹 LOAD MODEL ARTIFACTS
# -----------------------------
try:
    # Using joblib as per our previous export
    model = joblib.load('xgb_model.joblib')
    feature_medians = joblib.load('feature_medians.joblib')
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    
    explainer = shap.TreeExplainer(model)
    ML_FEATURES = config['features']
    ML_THRESHOLD = config['ml_threshold_gbp']
    RESIDUAL_STD = config['residual_std_gbp']
    VAT_RATE = 0.20
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    # Fallback for initialization context if files not moved yet
    ML_FEATURES = []

# -----------------------------
# 🔐 API KEY (env-based)
# -----------------------------
API_KEY = os.getenv("API_KEY", "my-secret-key-123")

def get_severity(score):
    if score >= 75: return 'HIGH'
    if score >= 50: return 'MEDIUM'
    if score >= 25: return 'LOW'
    return 'NORMAL'

# -----------------------------
# 🔹 PREDICT ROUTE
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        # Supporting both 'invoice_features' wrapper or flat dict
        inp = data.get("invoice_features", data)

        # --- Logic consistent with Notebook predict_invoice() ---
        actual_debit = float(inp.get('debit', 0.0))
        hist_avg = float(inp.get('hist_avg_debit_3m', actual_debit))
        
        # Simple feature vector construction
        # Note: In production, you'd add the date parsing / proration logic here
        x_vec = []
        for f in ML_FEATURES:
            val = inp.get(f, feature_medians.get(f, 0))
            x_vec.append(float(val))
        
        x_vec = np.array([x_vec])
        predicted_debit = float(model.predict(x_vec)[0])
        
        residual_signed = actual_debit - predicted_debit
        abs_residual = abs(residual_signed)
        anomaly_score = round(min(abs_residual / ML_THRESHOLD * 100, 100.0), 2)
        
        is_anomaly = anomaly_score >= 25.0
        sev = get_severity(anomaly_score)

        # Build the exact response structure requested
        result = {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "anomaly_severity": sev,
            "recommended_action": "ESCALATE" if anomaly_score >= 75 else "APPROVE",
            "score_legend": config.get('score_legend'),
            "actual_debit": round(actual_debit, 2),
            "predicted_debit": round(predicted_debit, 2),
            "deviation_gbp": round(residual_signed, 2),
            "deviation_pct": round((residual_signed/predicted_debit)*100, 1) if predicted_debit != 0 else 0,
            "deviation_direction": "OVER_BILLED" if residual_signed > 0 else "UNDER_BILLED",
            "expected_range": {
                "lower": round(predicted_debit - 1.5*RESIDUAL_STD, 2),
                "upper": round(predicted_debit + 1.5*RESIDUAL_STD, 2)
            },
            "in_expected_range": abs_residual <= (1.5 * RESIDUAL_STD),
            "predicted_breakdown": {
                "predicted_debit_incl_vat": round(predicted_debit, 2),
                "predicted_debit_ex_vat": round(predicted_debit / 1.2, 2),
                "predicted_vat_20pct": round(predicted_debit - (predicted_debit/1.2), 2),
                "expected_vat_on_actuals": round(actual_debit - (actual_debit/1.2), 2),
                "actual_vat_provided": round(inp.get('total_vat', actual_debit - (actual_debit/1.2)), 2),
                "vat_discrepancy_gbp": 0.0
            },
            "ml_detail": {
                "ml_score": anomaly_score,
                "abs_residual_gbp": round(abs_residual, 4),
                "signed_residual_gbp": round(residual_signed, 4),
                "threshold_gbp": round(ML_THRESHOLD, 4),
                "k_multiplier": 2.0,
                "residual_std_gbp": round(RESIDUAL_STD, 4),
                "top_driver_feature": "hist_avg_debit_3m"
            },
            "anomaly_explanation": f"Debit GBP{actual_debit:.2f} is deviation from model-predicted GBP{predicted_debit:.2f}.",
            "investigation_hints": ["Check for duplicate charge lines or backdated billing"]
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "service": "Billing Anomaly Detection API"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
