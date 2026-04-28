from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import json
import pandas as pd
from datetime import datetime, date
import xgboost as xgb
import shap
import logging

# ----------------------------
# App Init
# ----------------------------
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# ----------------------------
# Load Artifacts
# ----------------------------
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

# ----------------------------
# Config
# ----------------------------
API_KEY = os.getenv("API_KEY")  # NO DEFAULT
ML_FEATURES = config["features"]  # :contentReference[oaicite:1]{index=1}
ML_THRESHOLD = config["ml_threshold_gbp"]
RESIDUAL_STD = config["residual_std_gbp"]
ANOMALY_THRESHOLD = config["anomaly_threshold"]
VAT_RATE = config["vat_rate"]
CYCLE_MAP = {int(k): v for k, v in config["cycle_map"].items()}
SCORE_LEGEND = config["score_legend"]

# Optional SHAP
USE_SHAP = True
explainer = shap.TreeExplainer(model) if USE_SHAP else None

# ----------------------------
# Helpers
# ----------------------------
def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except:
        return default

def parse_date(v):
    try:
        return pd.to_datetime(v)
    except:
        return None

def get_severity(score):
    if score >= 75: return "HIGH"
    if score >= 50: return "MEDIUM"
    if score >= 25: return "LOW"
    return "NORMAL"

# ----------------------------
# Health Check
# ----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})

# ----------------------------
# Prediction Endpoint
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # API KEY VALIDATION
        if API_KEY and request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json(force=True)
        inp = data.get("invoice_features", data)

        # ----------------------------
        # Feature Engineering
        # ----------------------------
        bill_from = parse_date(inp.get("bill_from_date"))
        bill_thru = parse_date(inp.get("bill_thru_date"))

        actual_debit = safe_float(inp.get("debit"))
        credit = safe_float(inp.get("credit"))

        bal_fwd_raw = inp.get("bal_fwd")
        bal_fwd = safe_float(bal_fwd_raw) if bal_fwd_raw else None
        bal_fwd_filled = 0.0 if bal_fwd is None else bal_fwd

        is_new_account = bal_fwd is None

        pt_no = int(safe_float(inp.get("payment_terms_no", 90)))
        notify = int(safe_float(inp.get("notify_method", 3)))

        is_paper = int(notify == 10)
        cycle_d = CYCLE_MAP.get(pt_no, 90)

        if bill_from is not None and bill_thru is not None:
            billing_days = (bill_thru - bill_from).days + 1
            proration_factor = billing_days / cycle_d
        else:
            billing_days = feature_medians["billing_days"]
            proration_factor = feature_medians["proration_factor"]

        billing_month = bill_from.month if bill_from else feature_medians["billing_month"]

        hist_avg = safe_float(inp.get("hist_avg_debit_3m"), actual_debit)
        hist_std = safe_float(inp.get("hist_std_debit_3m"), 0.0)

        raw_features = {
            "billing_days": billing_days,
            "cycle_days": cycle_d,
            "proration_factor": proration_factor,
            "is_paper_bill": is_paper,
            "hist_avg_debit_3m": hist_avg,
            "hist_std_debit_3m": hist_std,
            "bal_fwd_filled": bal_fwd_filled,
            "billing_month": billing_month,
        }

        x_vec = np.array([[raw_features[f] for f in ML_FEATURES]])

        # ----------------------------
        # Prediction
        # ----------------------------
        predicted = float(model.predict(x_vec)[0])

        residual = actual_debit - predicted
        abs_residual = abs(residual)

        anomaly_score = min((abs_residual / ML_THRESHOLD) * 100, 100)
        is_anomaly = anomaly_score >= ANOMALY_THRESHOLD

        severity = get_severity(anomaly_score)

        # ----------------------------
        # SHAP
        # ----------------------------
        shap_output = {}
        top_driver = None

        if USE_SHAP:
            shap_vals = explainer.shap_values(x_vec)[0]
            shap_output = {
                f: float(v) for f, v in zip(ML_FEATURES, shap_vals)
            }
            top_driver = max(shap_output, key=lambda k: abs(shap_output[k]))

        # ----------------------------
        # Response
        # ----------------------------
        return jsonify({
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 2),
            "severity": severity,

            "actual": round(actual_debit, 2),
            "predicted": round(predicted, 2),
            "residual": round(residual, 2),

            "top_driver_feature": top_driver,
            "shap_values": shap_output,

            "score_legend": SCORE_LEGEND
        })

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)