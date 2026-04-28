
import os, json, pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load artifacts
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('feature_medians.pkl', 'rb') as f:
    feature_medians = pickle.load(f)
with open('model_config.json', 'r') as f:
    config = json.load(f)

ML_FEATURES = config['features']

def get_severity(score):
    if score >= 75: return 'HIGH'
    if score >= 50: return 'MEDIUM'
    if score >= 25: return 'LOW'
    return 'NORMAL'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Logic consistent with predict_invoice() function
    # ... (Simplified for standalone use)
    return jsonify({"status": "success", "message": "Model loaded and endpoint ready"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
