# These are the tools we use to measure how good our models are
from sklearn.metrics import (
    accuracy_score,      # percentage of correct predictions
    precision_score,     # of all patients we predicted positive, how many actually were?
    recall_score,        # of all actual positive patients, how many did we catch?
    f1_score,            # a single number that balances precision and recall
    confusion_matrix     # a grid showing exactly where the model got things right and wrong
)

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config
from dataset import load_data, get_features_and_target
from features import split_data, scale_features


def load_model(model_name):
    # Load the saved .pkl file back into memory
    # This gives us back the dictionary we saved in train.py
    # containing both the model and the scaler
    import os
    path = os.path.join(config.MODEL_DIR, f"{model_name}.pkl")
    saved = joblib.load(path)
    return saved['model'], saved['scaler']


def evaluate_model(model, X_test, y_test, model_name):
    # Ask the model to make predictions on the test set
    # X_test is the 61 patients the model has never seen
    y_pred = model.predict(X_test)

    # Calculate all our metrics by comparing y_pred (what the model said)
    # against y_test (what the correct answers actually are)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {accuracy:.2%}")   # e.g. 0.8524 → 85.24%
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")

    return y_pred, accuracy, f1


def baseline_accuracy(y_test):
    # The naive baseline: always predict the most common class
    # If we just said "everyone has heart disease" what accuracy would we get?
    # This is the minimum bar our model needs to beat to be useful
    most_common = y_test.value_counts().idxmax()
    baseline = (y_test == most_common).mean()
    print(f"\n--- Naive Baseline (always predict {most_common}) ---")
    print(f"Accuracy: {baseline:.2%}")
    return baseline


def plot_confusion_matrix(y_test, y_pred, model_name):
    # A confusion matrix is a 2x2 grid showing:
    # Top left:     True Negatives  — predicted 0, actually 0 (correct)
    # Top right:    False Positives — predicted 1, actually 0 (wrong — false alarm)
    # Bottom left:  False Negatives — predicted 0, actually 1 (wrong — missed a case)
    # Bottom right: True Positives  — predicted 1, actually 1 (correct)
    # In medicine, False Negatives are the most dangerous —
    # telling someone they're healthy when they're not
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,         # show the numbers inside each cell
        fmt='d',            # format as integers not decimals
        cmap='Blues',       # color scheme
        xticklabels=['No Disease', 'Disease'],
        yticklabels=['No Disease', 'Disease']
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_confusion_matrix.png')
    print(f"Confusion matrix saved to models/{model_name}_confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, model_name):
    # Random Forest gives us feature importances — a score for each feature
    # showing how much it contributed to the model's decisions
    # Logistic Regression gives us coefficients instead — we take the absolute
    # value because a large negative coefficient is just as important as a large positive one
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        importances = model.feature_importances_
    else:
        # Logistic Regression
        importances = np.abs(model.coef_[0])

    # Pair each feature name with its importance score and sort them
    features = config.FEATURE_COLUMNS
    pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    names, scores = zip(*pairs)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(scores), y=list(names))
    plt.title(f'Feature Importance — {model_name}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_feature_importance.png')
    print(f"Feature importance saved to models/{model_name}_feature_importance.png")
    plt.close()


if __name__ == '__main__':
    # Step 1 — Reload the data and recreate the exact same train/test split
    df = load_data()
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 2 — Print the naive baseline so we know the minimum bar to beat
    baseline_accuracy(y_test)

    # Step 3 — Evaluate both models
    for model_name in ['logistic_regression', 'random_forest']:
        # Load the saved model and scaler
        model, scaler = load_model(model_name)

        # Scale the test set using the loaded scaler
        X_test_scaled = scaler.transform(X_test)

        # Evaluate and print metrics
        y_pred, accuracy, f1 = evaluate_model(model, X_test_scaled, y_test, model_name)

        # Generate charts
        plot_confusion_matrix(y_test, y_pred, model_name)
        plot_feature_importance(model, model_name)
