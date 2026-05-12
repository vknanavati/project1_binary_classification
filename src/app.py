from flask import Flask, request, jsonify
import config
from predict import predict

# Create the Flask app — same as project 0
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # request.json reads the JSON body that was sent to this endpoint
    # For example:
    # {
    #   "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
    #   "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    #   "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    # }
    data = request.json

    # Basic validation — make sure the request actually has data
    if not data:
        # 400 means "Bad Request" — the caller sent something invalid
        return jsonify({'error': 'No data provided'}), 400

    # Check that all 13 required features are present in the request
    # If someone sends a request missing 'chol' for example, we catch it here
    missing = [col for col in config.FEATURE_COLUMNS if col not in data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Get the model name from the request, default to random_forest if not specified
    # This lets the caller choose which model to use:
    # {"model": "logistic_regression", "age": 63, ...}
    model_name = data.get('model', 'random_forest')

    # Remove the model key from data before passing to predict()
    # because predict() only expects the 13 feature columns
    patient_data = {k: v for k, v in data.items() if k != 'model'}

    # Call our predict function from predict.py
    result = predict(patient_data, model_name=model_name)

    # jsonify converts the Python dictionary into a JSON response
    # 200 means "OK" — everything worked
    return jsonify(result), 200


@app.route('/health', methods=['GET'])
def health():
    # A simple endpoint to check if the server is running
    # Useful for monitoring — you can ping /health to see if the API is up
    return jsonify({'status': 'ok'}), 200


if __name__ == '__main__':
    # debug=True means Flask will auto-reload when you change the code
    # and show detailed error messages — only use this in development
    print("Starting Heart Disease Prediction API...")
    print("POST to http://localhost:5001/predict with patient data")
    app.run(debug=True, host='0.0.0.0', port=5001)
