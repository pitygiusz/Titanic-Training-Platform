import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, tune=False):
    if tune:
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [5, 10]
        }
        pipeline = Pipeline(steps=[
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)
        model.fit(X_train, y_train)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, tune=False):
    if tune:
        param_grid = {
            'n_estimators':[50, 100],
            'max_depth':[3, 5],
            'learning_rate':[0.1]
        }
        xgb = XGBClassifier(random_state=42, eval_metric='logloss')
        cv = StratifiedKFold(n_splits=3, shuffle=True)
        model = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        model.fit(X_train, y_train)
    else:
        model = XGBClassifier(eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    return accuracy, report, cm, precision, recall, f1