# Project Structure

project1_heart_disease/
├── data/               ← dataset goes here
├── models/             ← saved model
├── src/
│   ├── config.py       ← settings
│   ├── dataset.py      ← load and clean the data
│   ├── features.py     ← prepare features for the model
│   ├── train.py        ← train the model
│   ├── evaluate.py     ← test accuracy, show results
│   └── predict.py      ← make predictions on new patients
├── requirements.txt
└── .gitignore