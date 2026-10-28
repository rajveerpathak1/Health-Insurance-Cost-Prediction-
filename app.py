from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS

from src.modeling import load_model, predict_one, MODEL_PATH

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Allow frontend to communicate with backend

# Load the trained model (a full sklearn Pipeline: encoding + regressor)
model = load_model(MODEL_PATH)

VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGION = {"northeast", "northwest", "southeast", "southwest"}


# Preprocessing / validation function
def preprocess_input(data):
    try:
        age = float(data.get("age", 0))
        bmi = float(data.get("bmi", 0))
        children = int(data.get("children", 0))
    except (TypeError, ValueError):
        raise ValueError("Invalid input data format")
    sex = str(data.get("sex", "")).lower()
    smoker = str(data.get("smoker", "")).lower()
    region = str(data.get("region", "northeast")).lower()
    if not (18 <= age <= 100):
        raise ValueError("age must be between 18 and 100")
    if not (10 <= bmi <= 70):
        raise ValueError("bmi must be between 10 and 70")
    if not (0 <= children <= 10):
        raise ValueError("children must be between 0 and 10")
    if sex not in VALID_SEX or smoker not in VALID_SMOKER or region not in VALID_REGION:
        raise ValueError("sex must be male/female, smoker yes/no, region one of " + ", ".join(sorted(VALID_REGION)))
    return dict(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        features = preprocess_input(data)
        prediction = predict_one(model, **features)

        return jsonify({"predicted_cost": round(prediction, 2)})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "Prediction failed: " + str(e)}), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
