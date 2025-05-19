import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, roc_curve, auc
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(data_path):
    """
    Preprocess the Titanic dataset with advanced feature engineering.
    
    Args:
        data_path (str): Path to the Titanic dataset CSV file.
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets for training and testing.
    """
    print("\n===== PREPROCESSING STEPS =====\n")
    
    # Load the dataset
    print("1. Loading the dataset...")
    df = pd.read_csv(data_path)
    print(f"   - Dataset shape: {df.shape}")
    print(f"   - Columns: {', '.join(df.columns.tolist())}")
    
    # Extract titles from names for feature engineering
    print("\n2. Feature Engineering - Extracting titles from names...")
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    df['Title'] = df['Title'].map(lambda x: title_mapping.get(x, 'Rare'))
    print(f"   - Extracted titles: {df['Title'].value_counts().to_dict()}")
    
    # Create family size feature
    print("\n3. Feature Engineering - Creating family size feature...")
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves
    print(f"   - Family size range: {df['FamilySize'].min()} to {df['FamilySize'].max()}")
    
    # Create is_alone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    print(f"   - Passengers traveling alone: {df['IsAlone'].sum()} ({df['IsAlone'].sum()/len(df)*100:.2f}%)")
    
    # Extract deck from cabin
    print("\n4. Feature Engineering - Extracting deck from cabin...")
    df['Deck'] = df['Cabin'].str.slice(0, 1)
    df['Deck'] = df['Deck'].fillna('U')  # Unknown
    print(f"   - Deck distribution: {df['Deck'].value_counts().to_dict()}")
    
    # Fare per person
    print("\n5. Feature Engineering - Creating fare per person feature...")
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # Age bands
    print("\n6. Feature Engineering - Creating age bands...")
    df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teenager', 'YoungAdult', 'Adult', 'Senior'])
    
    # Check for missing values
    print("\n7. Checking for missing values...")
    missing_values = df.isnull().sum()
    for col, count in missing_values.items():
        if count > 0:
            print(f"   - {col}: {count} missing values ({count/len(df)*100:.2f}%)")
    
    # Advanced imputation for Age using Title
    print("\n8. Advanced imputation for Age using Title...")
    # Fill missing ages with median age by title
    title_age_median = df.groupby('Title')['Age'].median()
    for title in title_age_median.index:
        df.loc[(df['Age'].isnull()) & (df['Title'] == title), 'Age'] = title_age_median[title]
    
    # If any ages are still missing, use the global median
    if df['Age'].isnull().sum() > 0:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    
    print(f"   - Remaining missing Age values: {df['Age'].isnull().sum()}")
    
    # Drop unnecessary columns
    print("\n9. Dropping unnecessary columns...")
    print(f"   - Dropping: PassengerId, Name, Ticket, Cabin")
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Extract target variable
    print("\n10. Extracting target variable...")
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target shape: {y.shape}")
    print(f"   - Class distribution: {y.value_counts().to_dict()}")
    
    # Split the data
    print("\n11. Splitting the data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   - Training set: {X_train.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    
    # Define preprocessing for numerical features
    print("\n12. Setting up preprocessing pipelines...")
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']
    print(f"   - Numerical features: {', '.join(numerical_features)}")
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    print(f"   - Numerical pipeline: KNN imputation → Standard scaling")
    
    # Define preprocessing for categorical features
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Deck', 'IsAlone', 'AgeBand']
    print(f"   - Categorical features: {', '.join(categorical_features)}")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    print(f"   - Categorical pipeline: Most frequent imputation → One-hot encoding")
    
    # Combine preprocessing steps
    print("\n13. Combining preprocessing steps...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit the preprocessor on the training data
    print("\n14. Applying preprocessing transformations...")
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    print(f"   - Preprocessed training data shape: {X_train_preprocessed.shape}")
    print(f"   - Preprocessed test data shape: {X_test_preprocessed.shape}")
    
    # Save the preprocessor for later use
    print("\n15. Saving the preprocessor...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print(f"   - Preprocessor saved to: models/preprocessor.pkl")
    
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor

def train_model(data_path):
    """
    Train an ensemble of models on the Titanic dataset with hyperparameter tuning.
    
    Args:
        data_path (str): Path to the Titanic dataset CSV file.
        
    Returns:
        model: Trained model
        accuracy: Model accuracy on test set
    """
    # Preprocess the data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(data_path)
    
    print("\n===== MODEL TRAINING =====\n")
    
    # Create cross-validation strategy
    print("1. Setting up cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train multiple models and evaluate with cross-validation
    print("\n2. Training and evaluating multiple models with cross-validation...")
    
    # Random Forest
    print("   - Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print(f"     Best parameters: {rf_grid.best_params_}")
    print(f"     Best CV accuracy: {rf_grid.best_score_:.4f}")
    rf_best = rf_grid.best_estimator_
    
    # Gradient Boosting
    print("\n   - Training Gradient Boosting...")
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 4],
        'subsample': [0.8, 1.0]
    }
    gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='accuracy', n_jobs=-1)
    gb_grid.fit(X_train, y_train)
    print(f"     Best parameters: {gb_grid.best_params_}")
    print(f"     Best CV accuracy: {gb_grid.best_score_:.4f}")
    gb_best = gb_grid.best_estimator_
    
    # AdaBoost
    print("\n   - Training AdaBoost...")
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    ada = AdaBoostClassifier(estimator=dt, random_state=42)
    ada_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1.0]
    }
    ada_grid = GridSearchCV(ada, ada_params, cv=cv, scoring='accuracy', n_jobs=-1)
    ada_grid.fit(X_train, y_train)
    print(f"     Best parameters: {ada_grid.best_params_}")
    print(f"     Best CV accuracy: {ada_grid.best_score_:.4f}")
    ada_best = ada_grid.best_estimator_
    
    # Logistic Regression
    print("\n   - Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    lr_grid = GridSearchCV(lr, lr_params, cv=cv, scoring='accuracy', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    print(f"     Best parameters: {lr_grid.best_params_}")
    print(f"     Best CV accuracy: {lr_grid.best_score_:.4f}")
    lr_best = lr_grid.best_estimator_
    
    # Decision Tree
    print("\n   - Training Decision Tree...")
    dt = DecisionTreeClassifier(random_state=42)
    dt_params = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    dt_grid = GridSearchCV(dt, dt_params, cv=cv, scoring='accuracy', n_jobs=-1)
    dt_grid.fit(X_train, y_train)
    print(f"     Best parameters: {dt_grid.best_params_}")
    print(f"     Best CV accuracy: {dt_grid.best_score_:.4f}")
    dt_best = dt_grid.best_estimator_
    
    # Create Voting Classifier (Ensemble)
    print("\n3. Creating Voting Classifier (Ensemble)...")
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_best),
            ('gb', gb_best),
            ('ada', ada_best),
            ('dt', dt_best),
            ('lr', lr_best)
        ],
        voting='soft'  # Use predicted probabilities
    )
    
    # Train the ensemble
    print("\n4. Training the ensemble model...")
    voting_clf.fit(X_train, y_train)
    print("   - Ensemble training complete")
    
    # Make predictions
    print("\n5. Making predictions on test set...")
    y_pred = voting_clf.predict(X_test)
    
    # Evaluate the model
    print("\n===== MODEL EVALUATION =====\n")
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"1. Accuracy: {accuracy:.4f}")
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print(f"\n2. Precision: {precision:.4f}")
    print(f"3. Recall: {recall:.4f}")
    print(f"4. F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\n5. Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\n6. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    print("   - Confusion matrix saved to: results/confusion_matrix.png")
    
    # Plot ROC curves
    print("\n7. Plotting ROC curves...")
    plt.figure(figsize=(10, 8))
    
    # Individual models
    models = {
        'Random Forest': rf_best,
        'Gradient Boosting': gb_best,
        'AdaBoost': ada_best,
        'Decision Tree': dt_best,
        'Logistic Regression': lr_best,
        'Ensemble': voting_clf
    }
    
    colors = ['blue', 'green', 'red', 'purple', 'cyan', 'black']
    for i, (name, model) in enumerate(models.items()):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_curves.png')
    print("   - ROC curves saved to: results/roc_curves.png")
    
    # Feature importance (using Random Forest as it has feature_importances_)
    print("\n8. Feature Importance (from Random Forest):")
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'cat':
            # Get the feature names from the one-hot encoder
            encoder = transformer.named_steps['onehot']
            encoded_features = encoder.get_feature_names_out(features)
            feature_names.extend(encoded_features.tolist())
        else:
            feature_names.extend(features)
    
    # Sort feature importances
    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("   Top 10 features:")
    for i in range(min(10, len(feature_names))):
        if i < len(indices):
            idx = indices[i]
            if idx < len(feature_names):
                print(f"   - {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(min(10, len(indices))), 
            [importances[i] for i in indices[:10]], 
            align="center")
    plt.xticks(range(min(10, len(indices))), 
               [feature_names[i] if i < len(feature_names) else "Unknown" for i in indices[:10]], 
               rotation=90)
    plt.tight_layout()
    plt.savefig('results/feature_importance.png')
    print("   - Feature importance plot saved to: results/feature_importance.png")
    
    # Save the ensemble model
    print("\n9. Saving the trained ensemble model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(voting_clf, 'models/titanic_ensemble_model.pkl')
    print(f"   - Ensemble model saved to: models/titanic_ensemble_model.pkl")
    
    # Also save individual models
    for name, model in models.items():
        if name != 'Ensemble':
            model_filename = 'models/titanic_' + name.lower().replace(" ", "_") + '_model.pkl'
            joblib.dump(model, model_filename)
            print(f"   - {name} model saved to: {model_filename}")
    
    return voting_clf, accuracy

if __name__ == "__main__":
    data_path = "data/Titanic-Dataset.csv"
    model, accuracy = train_model(data_path)
    print("\n===== SUMMARY =====\n")
    print(f"Model trained and saved with accuracy: {accuracy:.4f}")
    print(f"Model file: models/titanic_model.pkl")
    print(f"Preprocessor file: models/preprocessor.pkl")
    print(f"Results directory: results/")
    print("\nCompleted successfully!")

