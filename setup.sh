#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install setuptools first
pip install --upgrade pip
pip install setuptools wheel

# Install dependencies one by one
pip install numpy
pip install pandas
pip install scikit-learn
pip install joblib
pip install xgboost
pip install matplotlib
pip install seaborn

# Run the model
python model.py

# Deactivate the virtual environment when done
deactivate
