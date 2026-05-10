# Pandas is our data manipulation library — it loads the CSV into a DataFrame (a table)
import pandas as pd

# Import our settings file so we can use DATA_PATH, FEATURE_COLUMNS, etc.
import config


def load_data():
    # This function's only job is to read the CSV file and return it as a DataFrame

    # pd.read_csv() opens the file at DATA_PATH and turns it into a table with
    # named columns and numbered rows — this is called a DataFrame, stored in 'df'
    df = pd.read_csv(config.DATA_PATH)

    # df.shape is a tuple like (303, 14) — rows and columns
    # df.shape[0] is the number of rows, df.shape[1] is the number of columns
    # the f"..." syntax lets you embed variables directly inside a string
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Send the DataFrame back to whoever called this function
    return df


def check_data(df):
    # This function's job is to print a health check of the data so we can
    # spot problems before training — like missing values or imbalanced classes

    print("\n--- Data Summary ---")

    # df.shape prints the full (rows, columns) tuple
    print(f"Shape: {df.shape}")

    # df.dtypes lists each column and what type of data it holds
    # (int64 = whole numbers, float64 = decimals, object = text)
    print(f"\nColumn types:\n{df.dtypes}")

    # df.isnull() returns True/False for every cell — True if the value is missing
    # .sum() counts the Trues per column, giving us a missing value count per column
    # Missing values would be a problem — the model can't learn from blank cells
    print(f"\nMissing values:\n{df.isnull().sum()}")

    # value_counts() counts how many 1s and 0s are in the target column
    # This tells us if the dataset is balanced (roughly equal 1s and 0s)
    # or imbalanced (e.g. 90% 0s and 10% 1s), which affects how we evaluate the model
    print(f"\nTarget distribution:\n{df[config.TARGET_COLUMN].value_counts()}")


def get_features_and_target(df):
    # This function splits the table into two parts:
    # X = the input columns the model learns from (the 13 patient measurements)
    # y = the answer column the model is trying to predict (0 or 1)
    # By convention, inputs are called X (capital) and the target is called y (lowercase)

    # Select only the 13 columns listed in FEATURE_COLUMNS from config.py
    X = df[config.FEATURE_COLUMNS]

    # Select just the 'target' column — this is what we're predicting
    y = df[config.TARGET_COLUMN]

    # Return both — the caller gets X and y back as a pair
    return X, y


# This block only runs if you execute this file directly (python dataset.py)
# If another script imports dataset.py, this block is skipped
if __name__ == '__main__':

    # Load the CSV into a DataFrame
    df = load_data()

    # Print the health check summary
    check_data(df)

    # Split into features and target
    X, y = get_features_and_target(df)

    # X.shape tells us how many rows and columns are in the features table
    print(f"\nFeatures shape: {X.shape}")

    # y.shape tells us how many rows are in the target column (should match X rows)
    print(f"Target shape: {y.shape}")