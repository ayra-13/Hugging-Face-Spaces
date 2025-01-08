from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('student_score_predictor.pkl')

# Features for the model
activity_columns = ['30', '49', '100', '30.1', '15', '35', '45', '100.1', '32', '24', '40']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Extract provided features progressively
    provided_features = list(data.keys())
    missing_features = [col for col in activity_columns if col not in provided_features]
    
    # Ensure at least one feature is provided
    if not provided_features:
        return jsonify({"error": "No features provided. Provide at least one activity score."}), 400

    # Prepare input for the model
    feature_vector = [data.get(col, 0) for col in activity_columns]
    feature_df = pd.DataFrame([feature_vector], columns=activity_columns)
    
    # Predict final marks
    prediction = model.predict(feature_df)[0]

    return jsonify({
        "provided_features": provided_features,
        "missing_features": missing_features,
        "predicted_final_score": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
