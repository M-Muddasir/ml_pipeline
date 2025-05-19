import os
import pytest
import pandas as pd
import numpy as np
import joblib
from model import preprocess_data, train_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Define fixtures for reusable test components
@pytest.fixture(scope="module")
def data_path():
    """Fixture to provide the data path and ensure it exists."""
    path = "data/Titanic-Dataset.csv"
    assert os.path.exists(path), f"Test data file {path} not found"
    return path

@pytest.fixture(scope="module")
def preprocessed_data(data_path):
    """Fixture to provide preprocessed data for tests."""
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data_path)
    return X_train, X_test, y_train, y_test, preprocessor

@pytest.fixture(scope="module")
def raw_data(data_path):
    """Fixture to provide raw data for tests."""
    return pd.read_csv(data_path)

# Preprocessing tests
class TestPreprocessing:
    """Test cases for the data preprocessing functions."""
    
    def test_data_loading(self, raw_data):
        """Test that the dataset is loaded correctly."""
        # Check that the data is loaded correctly
        assert raw_data is not None, "Dataset should be loaded successfully"
        assert len(raw_data) > 0, "Dataset should not be empty"
        assert 'PassengerId' in raw_data.columns, "Dataset should contain PassengerId column"
        assert 'Survived' in raw_data.columns, "Dataset should contain Survived column"
    
    def test_preprocess_data_shapes(self, preprocessed_data):
        """Test that preprocessing returns correct data shapes."""
        X_train, X_test, y_train, y_test, preprocessor = preprocessed_data
        
        # Check that the returned data has the expected shapes
        assert X_train is not None, "X_train should not be None"
        assert X_test is not None, "X_test should not be None"
        assert y_train is not None, "y_train should not be None"
        assert y_test is not None, "y_test should not be None"
        
        # Check that the training and test sets have the expected split ratio (80/20)
        total_samples = len(y_train) + len(y_test)
        train_ratio = len(y_train) / total_samples
        test_ratio = len(y_test) / total_samples
        
        assert 0.75 <= train_ratio <= 0.85, f"Training set should be approximately 80% of the data, got {train_ratio:.2f}"
        assert 0.15 <= test_ratio <= 0.25, f"Test set should be approximately 20% of the data, got {test_ratio:.2f}"
        
        # Check that the preprocessor was created and saved
        assert preprocessor is not None, "Preprocessor should not be None"
        assert os.path.exists("models/preprocessor.pkl"), "Preprocessor should be saved to models/preprocessor.pkl"
    
    def test_feature_engineering(self, raw_data):
        """Test that feature engineering creates expected features."""
        # Manually perform some of the feature engineering steps
        df = raw_data.copy()
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        # Check that the feature engineering steps worked as expected
        assert 'Title' in df.columns, "Title feature should be created"
        assert 'FamilySize' in df.columns, "FamilySize feature should be created"
        assert 'IsAlone' in df.columns, "IsAlone feature should be created"
        
        # Check that the values are as expected
        assert df['FamilySize'].min() >= 1, f"Minimum family size should be at least 1, got {df['FamilySize'].min()}"
        assert df['IsAlone'].isin([0, 1]).all(), "IsAlone should only contain 0 or 1"
        
        # Verify that titles were extracted correctly
        common_titles = ['Mr', 'Mrs', 'Miss', 'Master']
        for title in common_titles:
            assert title in df['Title'].values, f"Common title '{title}' should be extracted"
        
        # Create a visualization of title distribution for the test report
        plt.figure(figsize=(10, 6))
        title_counts = df['Title'].value_counts()
        title_counts.plot(kind='bar')
        plt.title('Distribution of Passenger Titles')
        plt.xlabel('Title')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('results/title_distribution.png')
        plt.close()

# Model tests
class TestModel:
    """Test cases for the ML model."""
    
    @pytest.mark.slow
    def test_model_training(self, data_path):
        """Test that the model can be trained successfully."""
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Train the model
        model, accuracy = train_model(data_path)
        
        # Check that the model was created
        assert model is not None, "Model should be created successfully"
        assert os.path.exists("models/titanic_ensemble_model.pkl"), "Ensemble model should be saved"
        assert os.path.exists("models/titanic_random_forest_model.pkl"), "Random Forest model should be saved"
    
    @pytest.mark.slow
    def test_model_accuracy(self, data_path):
        """Test that the model achieves the expected accuracy."""
        # Train or load the model
        model, accuracy = train_model(data_path)
        
        # Check that accuracy is above 80%
        assert accuracy > 0.8, f"Model accuracy should be greater than 80%, but got {accuracy*100:.2f}%"
        
        # Create a visualization of model accuracy for the test report
        plt.figure(figsize=(8, 6))
        plt.bar(['Accuracy'], [accuracy], color='green')
        plt.axhline(y=0.8, color='red', linestyle='--', label='Minimum Required (80%)')
        plt.ylim(0, 1)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig('results/model_accuracy.png')
        plt.close()
    
    def test_model_files_exist(self):
        """Test that all expected model files exist."""
        expected_files = [
            "models/preprocessor.pkl",
            "models/titanic_ensemble_model.pkl",
            "models/titanic_random_forest_model.pkl",
            "models/titanic_gradient_boosting_model.pkl",
            "models/titanic_adaboost_model.pkl",
            "models/titanic_decision_tree_model.pkl",
            "models/titanic_logistic_regression_model.pkl"
        ]
        
        for file in expected_files:
            assert os.path.exists(file), f"Expected model file {file} does not exist"

# Custom test for feature importance
class TestFeatureImportance:
    """Test cases for feature importance analysis."""
    
    def test_feature_importance(self):
        """Test that feature importance can be extracted from the model."""
        # Check if the Random Forest model exists
        model_path = "models/titanic_random_forest_model.pkl"
        assert os.path.exists(model_path), f"Model file {model_path} not found"
        
        # Load the model
        rf_model = joblib.load(model_path)
        
        # Check that feature importance can be extracted
        assert hasattr(rf_model, 'feature_importances_'), "Random Forest model should have feature_importances_ attribute"
        
        # Check that feature importance is not empty
        assert len(rf_model.feature_importances_) > 0, "Feature importance should not be empty"
        
        # Check that the sum of feature importance is approximately 1
        assert 0.99 <= np.sum(rf_model.feature_importances_) <= 1.01, "Sum of feature importance should be approximately 1"

# Add conftest.py configuration for pytest-html
def pytest_html_report_title(report):
    report.title = "Titanic ML Pipeline Test Report"

# Add custom CSS to improve the report appearance
def pytest_configure(config):
    config._metadata['Project'] = 'Titanic ML Pipeline'
    config._metadata['Developer'] = 'MLOPs Team'
    config._metadata['Date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# Add hooks to include plots in the HTML report
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    
    # Add plots to the HTML report if they exist
    if report.when == "call":
        if "results/title_distribution.png" in os.listdir('results') if os.path.exists('results') else []:
            report.extra = [pytest.html.extras.image('results/title_distribution.png')]
        if "results/model_accuracy.png" in os.listdir('results') if os.path.exists('results') else []:
            report.extra = [pytest.html.extras.image('results/model_accuracy.png')]

# Main function to run tests with nice formatting
if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Run the tests with pytest
    pytest.main(['-v', '--html=test_report.html', '--self-contained-html'])
