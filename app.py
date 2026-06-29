import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables for model and columns
model = None
columns = None

def load_models():
    """Load model files lazily to avoid cold start issues"""
    global model, columns
    if model is None:
        model_path = os.path.join(BASE_DIR, "loan_default_model.pkl")
        columns_path = os.path.join(BASE_DIR, "model_columns.pkl")
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(columns_path):
            raise FileNotFoundError(f"Columns file not found at {columns_path}")
        
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
    
    return model, columns

@app.route("/")
def home():
    return jsonify({"message": "Loan Default Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load models
        model, columns = load_models()
        
        # Get input data
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        # Match training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        # Prediction
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

# For local development
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# For Vercel serverless deployment
app.debug = False