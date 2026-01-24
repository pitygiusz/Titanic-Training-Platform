import pandas as pd
import numpy as np
import re

def import_and_preprocess_data(filepath='train.csv'):
    """Imports data from a CSV file and performs initial preprocessing."""

    train = pd.read_csv(filepath)
    train['Sex'] = train['Sex'].map({'male':0,'female':1})
    train.fillna({'Embarked': train['Embarked'].mode()[0],
                'Age': train['Age'].median()}, inplace=True)
    train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
    return train

def feature_engineering(train):
    """Performs feature engineering on the dataset."""

    fe = train.copy()

    fe['Title'] = fe['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    rare_titles = ['Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms',
        'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess',
        'Jonkheer']
    fe['Title'] = fe['Title'].replace(rare_titles,'Rare')

    title_mapping = {"Mr":0, "Mrs":1, "Miss":2, "Rare":3 }
    fe['Title'] = fe['Title'].map(title_mapping)
    fe['Title'] = fe['Title'].fillna(0) 

    fe['FamilySize'] = fe['SibSp'] + fe['Parch'] + 1
    fe['IsAlone'] = 1*(fe['FamilySize']==1)

    fe['Cabin_letter'] = fe['Cabin'].astype(str).str[0]
    
    cabins = {'C':0, 'E':1, 'G':2, 'D':3, 'A':4, 'B':5, 'F':6, 'T':7, 'n':8}
    fe['Cabin_letter'] = fe['Cabin_letter'].map(cabins).fillna(8)

    return fe