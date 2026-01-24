import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


def train_logistic_regression(preprocessing_pipeline, X_train, y_train, param_grid=None, manual_params=None, scoring='accuracy'):
    """
    Train Logistic Regression with preprocessing pipeline.
    """
    if param_grid:
        # GridSearchCV mode
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_grid_updated = {f'classifier__{k}': v for k, v in param_grid.items()}
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid_updated, 
                                   cv=cv, scoring=scoring, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search
    else:
        # Manual parameters mode
        params = {**manual_params, 'random_state': 42}
        if 'max_iter' not in params:
            params['max_iter'] = 1000
        
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', LogisticRegression(**params))
        ])
        full_pipeline.fit(X_train, y_train)
        return full_pipeline


def train_random_forest(preprocessing_pipeline, X_train, y_train, param_grid=None, manual_params=None, scoring='accuracy'):
    """
    Train Random Forest with preprocessing pipeline.
    """
    if param_grid:
        # GridSearchCV mode
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, 
                                   cv=cv, scoring=scoring, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search
    else:
        # Manual parameters mode
        params = {**manual_params, 'random_state': 42}
        
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', RandomForestClassifier(**params))
        ])
        full_pipeline.fit(X_train, y_train)
        return full_pipeline


def train_xgboost(preprocessing_pipeline, X_train, y_train, param_grid=None, manual_params=None, scoring='accuracy'):
    """
    Train XGBoost with preprocessing pipeline.
    """
    if param_grid:
        # GridSearchCV mode
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
        
        param_grid_updated = {f'classifier__{k}': v for k, v in param_grid.items()}
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid_updated, 
                                   cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        return grid_search
    else:
        # Manual parameters mode
        params = {**manual_params, 'eval_metric': 'logloss', 'random_state': 42}
        
        full_pipeline = Pipeline([
            ('preprocessing', preprocessing_pipeline),
            ('classifier', XGBClassifier(**params))
        ])
        full_pipeline.fit(X_train, y_train)
        return full_pipeline