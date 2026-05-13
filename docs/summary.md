# Project 1 — Binary Classification Summary

## What We Built
A machine learning system that predicts whether a patient has heart disease
based on 13 medical measurements. The system includes two trained models,
an evaluation pipeline, and a live Flask API that accepts patient data and
returns a prediction.

---

## The Files and What Each One Does

config.py
  Stores all settings in one place — file paths, model parameters, and the
  list of feature columns. Every other script imports from here so there is
  one place to make changes.

dataset.py
  Loads heart.csv into a pandas DataFrame. Runs a health check printing
  shape, column types, missing values, and target distribution. Splits the
  table into X (13 features) and y (target column).

features.py
  Prepares the data for the model. Splits into 80% training and 20% test
  sets using train_test_split. Scales all 13 features to a common range
  using StandardScaler so no feature dominates due to its size.

train.py
  Trains two models on the 242 training patients and saves each one to disk
  as a .pkl file using joblib. Each .pkl contains both the trained model and
  the scaler as a dictionary.

evaluate.py
  Loads each saved model and runs it on the 61 test patients it has never
  seen. Prints accuracy, precision, recall, and F1. Saves a confusion matrix
  and feature importance chart for each model as .png files.

predict.py
  Loads a saved model, takes a single patient's measurements as a dictionary,
  scales them, and returns a prediction and probability. This is what the
  Flask API calls.

app.py
  Wraps predict.py in a Flask web server. POST to /predict with JSON patient
  data and get back a prediction. GET /health to check if the server is up.

---

## The Results

Naive Baseline (always predict majority class): 54.10%

Logistic Regression:
  Accuracy:  80.33%
  Precision: 76.92%
  Recall:    90.91%
  F1:        83.33%

Random Forest:
  Accuracy:  83.61%
  Precision: 78.05%
  Recall:    96.97%
  F1:        86.49%

Both models crushed the baseline. Random Forest won on every metric,
most importantly Recall — it caught 97% of actual heart disease cases,
missing almost none.

---

## Key Concepts Learned

Tabular data
  Structured data in rows and columns, like a spreadsheet. Each row is one
  patient, each column is one measurement. Very different from image data
  which is a grid of pixels.

Binary classification
  Predicting one of two outcomes — in this case 1 (heart disease) or 0
  (no heart disease). The model outputs a probability and we use 0.5 as
  the threshold to decide which class to assign.

Feature scaling
  Transforming all columns to a common range so no feature dominates due
  to the size of its numbers. Critical for Logistic Regression. We fit the
  scaler only on training data and apply the same transformation to test
  data and new predictions.

Train/test split
  Holding back 20% of the data so the model is evaluated on patients it
  has never seen. A model that scores well on training data but poorly on
  test data has memorized rather than learned — called overfitting.

Logistic Regression
  Finds the best set of weights for each feature and combines them into a
  scoring formula. Simple, fast, and explainable. Draws one straight
  decision boundary through the data.

Random Forest
  Builds 100 decision trees, each trained on a slightly different random
  sample of the data. Combines their votes for the final prediction.
  More powerful than Logistic Regression but less explainable.

Precision vs Recall
  Precision — of all patients flagged as sick, how many actually were?
  Recall — of all actually sick patients, how many did we catch?
  In medicine, Recall matters more. Missing a sick patient is more
  dangerous than a false alarm.

Confusion Matrix
  A 2x2 grid showing True Positives, True Negatives, False Positives,
  and False Negatives. More informative than accuracy alone because it
  shows exactly what kind of mistakes the model makes.

joblib
  Serializes trained Python objects to disk as .pkl files. Lets us train
  once and reuse the model forever without retraining.

Flask API
  Wraps the model in a web server so any application can send patient
  data via HTTP and receive a prediction back. This is how ML models
  get used in real products.

---

## The Most Important Thing to Remember
Complexity is not always better. Logistic Regression — one of the
simplest ML models — achieved 80% accuracy on a real medical dataset.
Random Forest only beat it by 3 percentage points despite being far
more complex. Always start with the simplest model and only add
complexity if the results justify it.