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

This is the file where the model actually learns — it takes the prepared data from features.py and trains two different models on it so we can compare them. We're trying two models intentionally: **Logistic Regression** and **Random Forest**.

**Logistic Regression's** job during training is to find the best weights for each feature so the scoring system is as accurate as possible.

The limitation: it can only draw one straight dividing line through the data. If the real pattern is more complicated and curvy, it struggles.

Imagine you're a doctor and you have data on 242 patients. You want to figure out a simple rule to predict heart disease. You notice that older patients with high cholesterol tend to have it more. So you come up with a scoring system:

score = (age × some weight) + (cholesterol × some weight) + (heart rate × some weight) + ...

Each feature gets multiplied by a weight — a number that represents how important that feature is. A high weight means that feature matters a lot. A negative weight means higher values actually push toward "no heart disease."
You add all those weighted values together and get one final score. If the score is above 0.5 → predict heart disease. Below 0.5 → predict no heart disease.

**Random Forests** Now imagine instead of one scoring formula, you ask 100 different doctors for their opinion — and each doctor asks a series of yes/no questions to reach their answer:
Doctor 1:
Is age over 55?
  YES → Is chest pain type 3?
    YES → Heart disease
    NO  → No heart disease
  NO → Is cholesterol over 240?
    YES → Heart disease
    NO  → No heart disease

Each doctor is a decision tree — a flowchart of yes/no questions. Each of the 100 doctors was trained on a slightly different random sample of patients, so they all learned slightly different question patterns.

When a new patient comes in, all 100 doctors make their prediction. If 67 say heart disease and 33 say no — the final answer is heart disease. Majority vote wins.

Why is this better than one doctor? Because one doctor might have learned some quirky pattern that doesn't generalize. But when 100 doctors all agree, you can be more confident.

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