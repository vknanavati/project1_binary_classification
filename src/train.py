# Two different model types we'll train and compare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# joblib saves the trained model to disk so we don't retrain every time
import joblib

# os lets us check if the models folder exists before saving
import os

# Import our own files
import config
from dataset import load_data, get_features_and_target
from features import split_data, scale_features


def train_logistic_regression(X_train, y_train):
    # Logistic Regression is a simple, fast model that draws a straight
    # decision boundary between classes
    # Analogy: imagine drawing one straight line through a scatter plot
    # to separate the dots into two groups — that's roughly what this does

    # max_iter=1000 means give it up to 1000 steps to find the best line
    # the default of 100 sometimes isn't enough and causes a warning
    model = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)

    # fit() is where the learning happens — the model looks at all 242
    # training patients and figures out the best decision boundary
    model.fit(X_train, y_train)

    print("Logistic Regression trained successfully")
    return model


def train_random_forest(X_train, y_train):
    # Random Forest builds many decision trees and combines their votes
    # Analogy: instead of asking one doctor for a diagnosis, you ask 100
    # doctors and go with the majority opinion — that's a Random Forest

    # n_estimators=100 means build 100 decision trees
    model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)

    # Same as before — fit() is where the learning happens
    model.fit(X_train, y_train)

    print("Random Forest trained successfully")
    return model


def save_model(model, scaler, model_name):
    # We save both the model AND the scaler together in one file
    # This is important — when predict.py gets a new patient, it needs
    # to scale their data the exact same way we scaled the training data
    # If we only saved the model, new predictions would be on unscaled
    # numbers and would be completely wrong

    # Build the save path — e.g. models/random_forest.pkl
    path = os.path.join(config.MODEL_DIR, f"{model_name}.pkl")

    # Save as a dictionary containing both objects
    joblib.dump({'model': model, 'scaler': scaler}, path)
    print(f"Model saved to {path}")


if __name__ == '__main__':
    # Step 1 — Load the raw data
    df = load_data()

    # Step 2 — Split into features and target
    X, y = get_features_and_target(df)

    # Step 3 — Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4 — Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 5 — Train both models
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train_scaled, y_train)

    # Step 6 — Save both models with their scaler
    save_model(lr_model, scaler, 'logistic_regression')
    save_model(rf_model, scaler, 'random_forest')

    print("\nAll models trained and saved successfully")
