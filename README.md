# Titanic ML Pipeline

![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/640px-RMS_Titanic_3.jpg)

This project implements a machine learning pipeline for the Titanic dataset. It includes functions for data preprocessing, feature engineering, model training, and model evaluation with a focus on achieving high accuracy.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Performance](#performance)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Contributing](#contributing)

## Features

- **Advanced Data Preprocessing**
  - Handling missing values using KNN imputation
  - Feature normalization with StandardScaler
  - One-hot encoding of categorical variables
  - Smart imputation strategies (e.g., age imputation based on passenger titles)

- **Feature Engineering**
  - Title extraction from passenger names
  - Family size calculation
  - IsAlone indicator for solo travelers
  - Deck information extraction from cabin
  - Fare per person calculation
  - Age bands for better classification

- **Ensemble Modeling with Multiple Algorithms**
  - Random Forest (84.13% CV accuracy)
  - Gradient Boosting (83.43% CV accuracy)
  - AdaBoost (83.71% CV accuracy)
  - Decision Tree (83.86% CV accuracy)
  - Logistic Regression (82.17% CV accuracy)
  - Voting Ensemble of all models

- **Advanced Model Training**
  - Cross-validation with StratifiedKFold
  - Hyperparameter tuning with GridSearchCV
  - Feature importance analysis

- **Comprehensive Evaluation**
  - Accuracy, Precision, Recall, F1 Score
  - Confusion matrix visualization
  - ROC curves for all models
  - Feature importance plots

## Project Structure

```
ml_pipeline/
├── data/
│   └── Titanic-Dataset.csv      # The Titanic dataset
├── models/                      # Directory for saved models
│   ├── preprocessor.pkl         # Saved preprocessor pipeline
│   ├── titanic_ensemble_model.pkl  # Saved ensemble model
│   └── titanic_*_model.pkl      # Individual model files
├── results/                     # Visualizations and results
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   ├── feature_importance.png   # Feature importance plot
│   └── roc_curves.png           # ROC curves for all models
├── .github/workflows/          # GitHub Actions workflows
│   └── ml_pipeline.yml          # CI/CD pipeline configuration
├── .gitignore                  # Git ignore file
├── model.py                    # Main ML pipeline code
├── test_model.py               # Unit tests
├── requirements.txt            # Project dependencies
├── setup.sh                    # Environment setup script
└── README.md                   # This file
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setting up the environment

1. Clone the repository
```bash
git clone https://github.com/yourusername/titanic-ml-pipeline.git
cd titanic-ml-pipeline
```

2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Alternatively, use the provided setup script:
```bash
chmod +x setup.sh
./setup.sh
```

## Usage

### Training the Model

To train and save the model:

```bash
python model.py
```

This will:
1. Load the Titanic dataset
2. Apply feature engineering and preprocessing
3. Train multiple models with hyperparameter tuning
4. Create an ensemble model
5. Evaluate the model performance
6. Generate visualizations
7. Save all models and preprocessor to the `models` directory

### Running Tests

To run the unit tests:

```bash
python -m pytest test_model.py -v
```

## Model Details

### Preprocessing Pipeline
- **Numerical Features**: Age, SibSp, Parch, Fare, FamilySize, FarePerPerson
  - KNN imputation for missing values
  - Standard scaling for normalization

- **Categorical Features**: Pclass, Sex, Embarked, Title, Deck, IsAlone, AgeBand
  - Most frequent imputation for missing values
  - One-hot encoding

### Models
- **Ensemble Model**: Voting Classifier with soft voting
- **Individual Models**:
  - Random Forest with optimized hyperparameters
  - Gradient Boosting with optimized hyperparameters
  - AdaBoost with optimized hyperparameters
  - Decision Tree with optimized hyperparameters
  - Logistic Regression with optimized hyperparameters

### Saved Files
- `models/preprocessor.pkl`: The preprocessing pipeline
- `models/titanic_ensemble_model.pkl`: The ensemble model
- `models/titanic_random_forest_model.pkl`: Random Forest model
- `models/titanic_gradient_boosting_model.pkl`: Gradient Boosting model
- `models/titanic_adaboost_model.pkl`: AdaBoost model
- `models/titanic_decision_tree_model.pkl`: Decision Tree model
- `models/titanic_logistic_regression_model.pkl`: Logistic Regression model

## Performance

The ensemble model achieves:
- **Accuracy**: 83.24%
- **Precision**: 83.12%
- **Recall**: 83.24%
- **F1 Score**: 83.08%

Detailed performance by class:
```
              precision    recall  f1-score   support

           0       0.84      0.89      0.87       110
           1       0.81      0.74      0.77        69

    accuracy                           0.83       179
   macro avg       0.83      0.82      0.82       179
weighted avg       0.83      0.83      0.83       179
```

## CI/CD Pipeline

This project includes a GitHub Actions workflow that automates the following steps:

1. **Environment Setup**
   - Sets up Python 3.10 with pip caching
   - Installs all dependencies including pytest and pytest-html

2. **Testing**
   - Runs all unit tests with pytest
   - Generates an HTML test report with visualizations
   - Uploads the test report as an artifact

3. **Model Training**
   - Trains the ML model with the latest code
   - Captures model output for reporting

4. **Artifact Management**
   - Saves all model files (.pkl) and visualizations (.png)
   - Generates a model performance report
   - Makes all artifacts available for download
   - Preserves artifacts for 14 days

The workflow runs on every push or pull request to the main/master branch, ensuring continuous integration and deployment of the ML pipeline.

### CI/CD Workflow File

The GitHub Actions workflow is defined in `.github/workflows/ml_pipeline.yml` and includes the following key components:

```yaml
name: Titanic ML Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
      - name: Set up Python
      - name: Install dependencies
      - name: Run unit tests
      - name: Upload test report
      - name: Train model
      - name: Upload model artifacts
      - name: Generate model report
      - name: Upload model report
```

## Testing

The project includes comprehensive unit tests using pytest with HTML reporting:

1. **Data Preprocessing**
   - Verifies correct data shapes
   - Ensures proper train/test split
   - Validates preprocessor creation and saving

2. **Feature Engineering**
   - Tests feature creation
   - Validates feature values
   - Visualizes title distribution

3. **Model Performance**
   - Ensures model accuracy exceeds 80%
   - Visualizes model accuracy against threshold
   - Verifies all model files are created

4. **Feature Importance**
   - Tests feature importance extraction
   - Validates importance values

### Running Tests

To run the tests with HTML report generation:

```bash
python -m pytest test_model.py -v --html=test_report.html --self-contained-html
```

This will generate a comprehensive HTML report with test results, visualizations, and metadata.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

Developed as part of MLOPs Assignment 2.
