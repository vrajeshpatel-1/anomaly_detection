
from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import json
import xgboost as xgb

app = Flask(__name__)

# Load artifacts
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
try:
    model.set_params(device="cpu", predictor="cpu_predictor")
except:
    pass

with open("feature_medians.pkl", "rb") as f:
    feature_medians = pickle.load(f)
with open("model_config.json", "r") as f:
    config = json.load(f)

API_KEY = os.getenv("API_KEY", "my-secret-key-123")

def get_severity(score):
    if score >= 75: return "HIGH"
    if score >= 50: return "MEDIUM"
    if score >= 25: return "LOW"
    return "NORMAL"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        raw_data = request.get_json(force=True)
        inp = raw_data.get("invoice_features", raw_data)

        # Build Feature Vector
        features = config["features"]
        x_vec = [float(inp.get(f, feature_medians.get(f, 0))) for f in features]

        # Inference
        prediction = float(model.predict(np.array([x_vec]))[0])
        actual = float(inp.get("debit", 0))
        residual = actual - prediction
        
        # Scoring Logic
        threshold = config["ml_threshold_gbp"]
        score = round(min(abs(residual) / threshold * 100, 100.0), 2)
        severity = get_severity(score)

        return jsonify({
            "is_anomaly": score >= 25.0,
            "anomaly_score": score,
            "anomaly_severity": severity,
            "recommended_action": "ESCALATE" if score >= 75 else "HOLD_FOR_REVIEW" if score >= 50 else "FLAG" if score >= 25 else "APPROVE",
            "actual_debit": round(actual, 2),
            "predicted_debit": round(prediction, 2),
            "deviation_gbp": round(residual, 2),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "v5 Active", "threshold": config["ml_threshold_gbp"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
