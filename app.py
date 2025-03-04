from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the saved model
model_path = "crop_recommendation_model_latest.h5"
model = load_model(model_path)

label_map = {
    0: "กล้วย",
    1: "กาแฟ",
    2: "ข้าว",
    3: "มะม่วง",
    4: "มะละกอ",
    5: "แตงโม",
}
label_encoder = LabelEncoder()


@app.route("/predict_batch", methods=["GET"])
def predict_batch():
    try:

        # Read the CSV data into a pandas DataFrame
        df = pd.read_csv("testset.csv")

        # Separate the features and labels
        X = df.drop(columns=["label"])
        y = df["label"]

        # Encode labels for comparison
        y_encoded = label_encoder.fit_transform(y)

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Predict using the model
        predictions = model.predict(X_scaled)
        predicted_labels = np.argmax(predictions, axis=1)

        # Map predicted labels to their respective class names
        predicted_names = [
            label_map.get(label, "Unknown") for label in predicted_labels
        ]

        # Count how many predictions match the actual labels
        correct_predictions = sum(
            predicted_names[i] == y.iloc[i] for i in range(len(y))
        )

        # Return the result
        result = {
            "total_predictions": len(y),
            "correct_predictions": correct_predictions,
            "accuracy": correct_predictions / len(y),
            "predictions_count": {
                label_map[label]: list(predicted_names).count(label_map[label])
                for label in label_map
            },
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON input data
        data = request.get_json()

        # Required feature keys
        required_keys = [
            "nitrogen",
            "phosphorus",
            "potassium",
            "ph",
            "humidity",
            "temperature",
            "rainfall",
        ]

        # Check if all required keys are in the request
        if not all(key in data for key in required_keys):
            return jsonify({"error": "Missing one or more required features"}), 400

        # Extract values in the correct order and convert to float
        features = np.array([float(data[key]) for key in required_keys]).reshape(1, -1)
        print(features)
        # Make the prediction
        predictions = model.predict(features)
        predicted_label = np.argmax(predictions, axis=1)[0]  # Get the predicted class

        # Define the label map
        label_map = {
            0: "กล้วย",
            1: "กาแฟ",
            2: "ข้าว",
            3: "มะม่วง",
            4: "มะละกอ",
            5: "แตงโม",
        }

        # Get the human-readable label
        predicted_name = label_map.get(predicted_label, "Unknown")

        # Return the prediction as a JSON response
        return jsonify({"predicted_name": predicted_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Define a route to check if the API is working
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "API is working"})


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
