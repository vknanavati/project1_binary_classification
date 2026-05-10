# config.py — What Each Setting Does

**`BASE_DIR`, `DATA_DIR`, `MODEL_DIR`**
Builds file paths that work on any computer regardless of where the project lives.
Instead of hardcoding something like `/Users/viminnanavati/project1/data`, it figures
out the path dynamically based on where the script is located. Same trick as project 0.

**`DATA_PATH`**
The full path to the dataset CSV file. Every script that needs to load the data
imports this instead of typing the path manually.

**`MODEL_PATH`**
Where the trained model gets saved after training. joblib will write a `.pkl` file
here, and predict.py will load it back from this same path.

**`TEST_SIZE = 0.2`**
We'll hold back 20% of the data to test the model on after training. The model
never sees this data during training, so it's a fair, unbiased test of how well
it learned.

**`RANDOM_STATE = 42`**
Many ML operations involve randomness — like shuffling the data before splitting it.
Setting this to a fixed number means you get the same result every time you run the
code. 42 is just a convention (a nerdy reference), any number works.

**`FEATURE_COLUMNS`**
The 13 patient measurements the model will use to make its prediction — things like
age, cholesterol, and heart rate. The model looks at all 13 and outputs a single
answer: has heart disease (1) or doesn't (0).