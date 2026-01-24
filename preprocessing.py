import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def engineer_features(X):
    """
    Feature engineering function - all transformations are deterministic.
    """

    X = X.copy()
    
    # Map Sex 
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    
    # Extract Title from Name 
    X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
                  'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer']
    X['Title'] = X['Title'].replace(rare_titles, 'Rare')
    title_mapping = {"Mr": 0, "Mrs": 1, "Miss": 2, "Rare": 3}
    X['Title'] = X['Title'].map(title_mapping).fillna(0)
    
    # Family features 
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X['IsAlone'] = (X['FamilySize'] == 1).astype(int)
    
    # Select final features
    feature_cols = ['Pclass', 'Sex', 'Age', 'Fare', 'Title', 'FamilySize', 'IsAlone', 'Embarked']
    X = X[feature_cols]
    
    return X


def load_data(filepath='train.csv'):
    """
    Load raw data without any preprocessing
    """
    return pd.read_csv(filepath)


def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline that includes feature engineering and preprocessing steps.
    """
    
    # Step 1: Feature engineering transformer
    feature_eng = FunctionTransformer(engineer_features, validate=False)
    
    # Step 2: Define columns for different preprocessing
    numeric_features = ['Age', 'Fare']
    categorical_features = ['Embarked']
    passthrough_features = ['Pclass', 'Sex', 'Title', 'FamilySize', 'IsAlone']
    
    # Step 3: Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ]), categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ],
        remainder='drop'
    )
    
    # Step 4: Combine into full pipeline
    full_pipeline = Pipeline([
        ('feature_engineering', feature_eng),
        ('preprocessing', preprocessor)
    ])
    
    return full_pipeline