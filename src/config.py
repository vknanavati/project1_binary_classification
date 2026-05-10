import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

DATA_PATH = os.path.join(DATA_DIR, 'heart.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'heart_model.pkl')

# --- Model Settings ---
TEST_SIZE = 0.2        # 20% of data used for testing, 80% for training
RANDOM_STATE = 42      # makes random operations reproducible
TARGET_COLUMN = 'target'  # the column we're trying to predict

# --- Features ---
# These are the columns the model will use to make predictions
FEATURE_COLUMNS = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
]
