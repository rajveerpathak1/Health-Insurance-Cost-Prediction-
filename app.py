from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
from pathlib import Path
from flask_cors import CORS

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Allow frontend to communicate with backend

# Load the trained model
model_path = Path("D:\Project\Health-Insurance-Cost-Prediction-\insurance_cost_model.pkl")
model = joblib.load(model_path)

# Preprocessing function
def preprocess_input(data):
    try:
        age = float(data.get('age', 0))
        bmi = float(data.get('bmi', 0))
        children = int(data.get('children', 0))
        
        # One-hot encoding
        sex_male = 1 if data.get('sex', '').lower() == 'male' else 0
        smoker_yes = 1 if data.get('smoker', '').lower() == 'yes' else 0
        
        # Encoding region as a single categorical variable
        region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
        region = region_mapping.get(data.get('region', '').lower(), 0)  # Default to 'northeast'
        
        return np.array([[age, bmi, children, sex_male, smoker_yes, region]])
    except ValueError:
        raise ValueError("Invalid input data format")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        input_features = preprocess_input(data)
        prediction = model.predict(input_features)[0]
        
        return jsonify({"predicted_cost": round(prediction, 2)})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
