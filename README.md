# Titanic-ML-Project

This project is based on the classic [Kaggle Titanic: Machine Learning from Disaster competition](https://www.kaggle.com/competitions/titanic).  
The goal is to **predict which passengers survived** the Titanic shipwreck using **machine learning models**.

### What I did:
- Loaded and explored the dataset  
- Performed feature engineering  
- Built and compared multiple models:
  - Logistic Regression (baseline)
  - XGBoost (with GridSearchCV hyperparameter tuning)
  - Random Forest (with GridSearchCV hyperparameter tuning)
- Created a simple ensemble by averaging model predictions  
- Evaluated models using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1 score

My ensemble model achieved an accuracy of **0.78229** on the test set with unknown ground truth.

Additionally, I added a simple neural network built using PyTorch. It achieved an accuracy of **0.75837**.

### Technologies used:
- Python (pandas, numpy)
- scikit-learn
- PyTorch

The project was completed individually.
