# Titanic-ML-Project

An interactive machine learning application built with Streamlit to predict passenger survival on the Titanic. This tool provides a user-friendly interface for training, tuning, and evaluating different classification models without touching the code.

**Live Demo:** [Titanic Training Platform](https://pitygiusz-titanic-ml-project.streamlit.app/)

## Goal

The primary goal of this project is to elevate the standard Titanic analysis by converting a usual Jupyter Notebook into a modular, interactive web application using Streamlit. 

## Key Features

- **Multi-Model Support:** Train and compare various algorithms: Logistic Regression, Random Forest, and XGBoost.

- **Two Tuning Modes:**

  - Manual Mode: Experiment with hyperparameters using interactive sliders.

  - GridSearch Mode: Automatically find the best parameters optimized for Accuracy, Precision, Recall, or F1-Score.

- **Comprehensive Evaluation:** Analyze model performance via Confusion Matrices, Classification Reports, and cross-validated metrics.

- **One-Click Submission:** Automatically process the raw test.csv using the trained pipeline and download a formatted submission.csv for Kaggle.

- **Robust Preprocessing:** Handles missing data and feature engineering dynamically behind the scenes to ensure reliable predictions.

## Project Structure
```
Titanic-ML-Project/
├── app.py                # Main Streamlit application 
├── preprocessing.py      # Feature engineering &  pipelines
├── training.py           # Model definitions and GridSearch setup
├── requirements.txt      # Project dependencies
├── train.csv             # Training dataset
├── test.csv              # Test dataset
├── first_look.ipynb      # Notebook with first look at data and modelling
└── README.md             # Documentation
```

## How to Run

1. Clone the repository

```Bash
git clone https://github.com/pitygiusz/Titanic-ML-Project
cd Titanic-ML-Project
```

2. Install dependencies

```Bash
pip install -r requirements.txt
```

3. Run the application

```Bash
streamlit run app.py
```

## Dataset

This project is based on the classic [Kaggle Titanic: Machine Learning from Disaster competition](https://www.kaggle.com/competitions/titanic).  


## Technologies Used
- Python (pandas, numpy)
- Streamlit
- Scikit-Learn, XGBoost
- PyTorch

## Contributions
This project was completed individually.
