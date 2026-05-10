# Project 1 — Libraries Reference

## What Each Library Does

**`pandas`**
Reads the raw dataset file (a CSV in our case) and loads it into a structure called a
DataFrame — which is just a table with named columns and numbered rows. Once it's loaded,
pandas lets you do things like: drop columns you don't need, fill in missing values, filter
rows, and rename columns. It's not converting anything — the data is already in rows and
columns in the CSV file. Pandas just gives you a Python object that lets you work with it
easily.

**`numpy`**
Handles the actual number crunching underneath everything else. When pandas stores a column
of numbers, it stores them as a numpy array internally. When scikit-learn trains a model,
it's doing matrix math using numpy. You won't call numpy directly very often in this project,
but if you ever do something like calculate an average or reshape a list of numbers, that's
numpy. Think of it as the calculator that everything else uses.

**`scikit-learn`**
The library that actually builds and trains the ML model. It takes your cleaned data, fits a
model to it (learning the patterns), and then uses that model to make predictions on new data.
It also has helper tools for splitting your data into training and test sets, scaling numbers
to a common range, and measuring how accurate your model is.

**`matplotlib` and `seaborn`**
Visualization libraries. After training, you want to see how the model performed — not just
as a number, but visually. We'll use these to plot a confusion matrix (a grid showing where
the model got things right vs wrong) and a feature importance chart (which patient details
mattered most to the prediction). Seaborn produces nicer-looking charts with less code;
matplotlib is the lower-level engine it runs on.

**`flask`**
Same as project 0 — once the model is trained and saved, Flask lets us wrap it in a small
web server. You send it a patient's data via an HTTP request, and it sends back a prediction.
This is how ML models get used in real products.

**`joblib`**
Saves the trained model to a file on disk (a .pkl file) so you don't have to retrain it
every time. Once saved, any script can load it back and use it instantly. In project 0,
PyTorch had its own way of saving models (.pt files) — joblib is the equivalent for
scikit-learn models.