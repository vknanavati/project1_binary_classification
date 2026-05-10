# Project 1 — Script Reference

## What Each Script Does

---

### `config.py`
The settings file. Stores file paths, model settings, and the list of feature columns.
Every other script imports this so there is one place to change settings. Nothing runs
when you execute this file — it just holds values.

---

### `dataset.py`
Loads the raw CSV file into a pandas DataFrame and runs a health check on it.
Has three functions:
- `load_data()` — reads heart.csv and returns it as a table
- `check_data()` — prints the shape, column types, missing values, and target distribution
- `get_features_and_target()` — splits the table into X (the 13 input columns)
  and y (the target column we're predicting)

---

### `features.py`
Takes the raw X and y from dataset.py and prepares them for the model.

This includes **scaling the numbers** to a common range (0-1) using StandardScaler and **splitting the data** into a training set and a test set.

The problem is that most ML models are sensitive to the size of numbers. If you feed it these raw numbers, the model might pay way more attention to chol (which has big numbers like 500) than to fbs (which is just 0 or 1), not because cholesterol is more important but simply because its numbers are bigger.
Scaling fixes this by transforming every column so it lives in the same range. The most common method is called StandardScaler, which transforms each column so that:

The average becomes 0
Most values fall between -3 and 3

So after scaling, cholesterol of 250 might become 0.3, and an age of 63 might become 1.2. The actual values don't matter anymore — what matters is where each value sits relative to the rest of its column.

---

### `train.py`
Builds the ML model, feeds it the training data, and lets it learn the patterns.
Saves the trained model to disk as a .pkl file using joblib so we don't have
to retrain every time.

---

### `evaluate.py`
Loads the saved model and runs it on the test set — data the model has never seen.
Prints accuracy, generates a confusion matrix, and shows which features mattered
most to the model's decisions.

---

### `predict.py`
Loads the saved model and uses it to make a prediction on a single new patient.
This is what the Flask API will call when someone sends it patient data.

---

### `app.py`
Wraps the model in a small web server using Flask. You send it a patient's
measurements via an HTTP request and it sends back a prediction — has heart
disease (1) or doesn't (0).