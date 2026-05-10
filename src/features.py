# StandardScaler is the tool that scales our numbers to a common range
# train_test_split is the tool that divides our data into training and test sets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import our settings
import config

def split_data(X, y):
    # Split X and y into four parts:
    # X_train — the features the model will learn from (80% of the data)
    # X_test  — the features we'll use to test the model (20% of the data)
    # y_train — the correct answers for the training set
    # y_test  — the correct answers for the test set
    # The model will ONLY see X_train and y_train during training
    # X_test and y_test are kept hidden until evaluation

    # test_size=0.2 means 20% goes to test, 80% goes to training
    # random_state makes the split the same every time you run the code
    # stratify=y makes sure the split has roughly equal 1s and 0s in both sets
    # without stratify, you might accidentally put most of the 1s in one set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    print(f"Training set: {X_train.shape[0]} patients")
    print(f"Test set:     {X_test.shape[0]} patients")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    # Create a StandardScaler object — this is the tool that does the scaling
    scaler = StandardScaler()

    # fit_transform does two things in one step:
    # fit    — the scaler looks at X_train and calculates the average and
    #          spread of each column
    # transform — it then uses those calculations to scale the numbers
    # We only fit on X_train, never on X_test — the test set should be
    # completely unknown to everything, including the scaler
    X_train_scaled = scaler.fit_transform(X_train)

    # transform (without fit) scales X_test using the same calculations
    # from X_train — we don't recalculate for X_test because in the real
    # world you won't have test data when you first train the model
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled successfully")

    # Return both the scaled data AND the scaler itself
    # We need to save the scaler later so predict.py can scale new patients
    # the exact same way
    return X_train_scaled, X_test_scaled, scaler


if __name__ == '__main__':
    # Import dataset functions so we can test this file on its own
    from dataset import load_data, get_features_and_target

    # Load the data
    df = load_data()

    # Split into features and target
    X, y = get_features_and_target(df)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Scale the features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print(f"\nFirst row before scaling: {X_train.iloc[0].values}")
    print(f"First row after scaling:  {X_train_scaled[0]}")
