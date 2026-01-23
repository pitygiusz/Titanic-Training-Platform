# Titanic-ML-Project

## Project Overview
This project is based on the classic [Kaggle Titanic: Machine Learning from Disaster competition](https://www.kaggle.com/competitions/titanic).  

## Goal
The goal was to **predict which passengers survived** the Titanic shipwreck using **machine learning models**.

## Project structure
- Loading and exploring the dataset  
- Performing feature engineering  
- Building and comparing multiple models:
  - Logistic Regression (baseline)
  - XGBoost (with GridSearchCV hyperparameter tuning)
  - Random Forest (with GridSearchCV hyperparameter tuning)
- Creating a simple ensemble by averaging model predictions  
- Evaluating models using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1 score

My ensemble model achieved an accuracy of **0.78229** on the test set with unknown ground truth.

Additionally, I added a simple neural network built using PyTorch. It achieved an accuracy of **0.75837**.

## Technologies Used
- Python (pandas, numpy)
- scikit-learn
- PyTorch

## Contributions
This project was completed individually.
