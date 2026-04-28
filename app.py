
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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        raw_data = request.get_json(force=True)
        # Robust logic: Check for nested 'invoice_features' OR use the flat root object
        inp = raw_data.get("invoice_features", raw_data)

        features = config["features"]
        x_vec = []
        for f in features:
            # Use provided value if available, else fallback to median
            val = inp.get(f, feature_medians.get(f, 0))
            x_vec.append(float(val))

        prediction = float(model.predict(np.array([x_vec]))[0])
        actual = float(inp.get("debit", 0))
        residual = actual - prediction
        score = round(min(abs(residual) / config["ml_threshold_gbp"] * 100, 100.0), 2)

        return jsonify({
            "is_anomaly": score >= 25.0,
            "anomaly_score": score,
            "actual_debit": round(actual, 2),
            "predicted_debit": round(prediction, 2),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "v3 Active", "mode": "Robust Preprocessing"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
