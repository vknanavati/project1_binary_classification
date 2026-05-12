import joblib
import numpy as np
import os
import config


def load_model(model_name):
    # Load the saved .pkl file back into memory
    # Same as what evaluate.py does — gives us back the model and scaler
    path = os.path.join(config.MODEL_DIR, f"{model_name}.pkl")
    saved = joblib.load(path)
    return saved['model'], saved['scaler']


def predict(patient_data, model_name='random_forest'):
    # patient_data is a dictionary of one patient's measurements
    # for example:
    # {
    #   'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145,
    #   'chol': 233, 'fbs': 1, 'restecg': 0, 'thalach': 150,
    #   'exang': 0, 'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    # }

    # Load the trained model and scaler from disk
    model, scaler = load_model(model_name)

    # Pull the feature values out of the dictionary in the exact same
    # order as FEATURE_COLUMNS in config.py — order matters here because
    # the model learned from columns in a specific order during training
    values = [patient_data[col] for col in config.FEATURE_COLUMNS]

    # Convert the list of values into a numpy array and reshape it
    # The model expects a 2D array (a table) even for one patient
    # reshape(1, -1) means "1 row, figure out the columns automatically"
    # without this reshape the model would throw an error
    patient_array = np.array(values).reshape(1, -1)

    # Scale the patient's data using the same scaler from training
    # This is critical — if we don't scale, the numbers are in the wrong
    # range and the prediction will be garbage
    patient_scaled = scaler.transform(patient_array)

    # Ask the model to make a prediction — returns 0 or 1
    prediction = model.predict(patient_scaled)[0]

    # predict_proba returns the probability for each class [prob_0, prob_1]
    # [0] gets the first patient (we only have one)
    # [1] gets the probability of class 1 (heart disease)
    probability = model.predict_proba(patient_scaled)[0][1]

    result = {
        'prediction': int(prediction),
        'label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
        'probability': round(float(probability), 4),
        'model_used': model_name
    }

    return result


if __name__ == '__main__':
    # Test with a sample patient — these values are from the first row
    # of our dataset so we know the correct answer is 1 (heart disease)
    sample_patient = {
        'age': 63,
        'sex': 1,
        'cp': 3,
        'trestbps': 145,
        'chol': 233,
        'fbs': 1,
        'restecg': 0,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 2.3,
        'slope': 0,
        'ca': 0,
        'thal': 1
    }

    print("--- Testing Logistic Regression ---")
    result = predict(sample_patient, model_name='logistic_regression')
    print(f"Prediction: {result['label']}")
    print(f"Probability: {result['probability']:.2%}")

    print("\n--- Testing Random Forest ---")
    result = predict(sample_patient, model_name='random_forest')
    print(f"Prediction: {result['label']}")
    print(f"Probability: {result['probability']:.2%}")
