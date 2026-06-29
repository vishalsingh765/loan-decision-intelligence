import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = None
columns = None

def load_models():
    global model, columns
    if model is None:
        model_path = os.path.join(BASE_DIR, "loan_default_model.pkl")
        columns_path = os.path.join(BASE_DIR, "model_columns.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing: {model_path}")
        if not os.path.exists(columns_path):
            raise FileNotFoundError(f"Columns file missing: {columns_path}")
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
    return model, columns

@app.route('/')
def home():
    # Serve the index.html file
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model, columns = load_models()
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=columns, fill_value=0)

        pred = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0].tolist() if hasattr(model, 'predict_proba') else None

        response = {"prediction": int(pred)}
        if pred_proba:
            response["probability"] = pred_proba

        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({"error": f"Model file error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For local development only – Vercel uses the WSGI entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
