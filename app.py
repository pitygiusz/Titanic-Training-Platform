import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


try:
    import preprocessing as pp
    import training as tr
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="Titanic ML Modular App", layout="wide")

# Helper function to load and cache data
@st.cache_data
def get_data():
    try:
        # 1. Load & Preprocess
        raw_data = pp.import_and_preprocess_data('train.csv')
        
        # 2. Feature Engineering
        processed_data = pp.feature_engineering(raw_data)
        
        # 3. Feature Selection
        features = ['Pclass','Sex','Age','Fare','Title','FamilySize','IsAlone','Embarked_Q','Embarked_S']
        
        valid_features = [f for f in features if f in processed_data.columns]
        
        X = processed_data[valid_features]
        y = processed_data['Survived']
        
        return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
    except FileNotFoundError:
        st.error("Source data did not found.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error while processing {e}")
        return None, None, None, None



st.title("Titanic: Modular Training Platform")


X_train, X_val, y_train, y_val = get_data()

if X_train is not None:
    st.sidebar.header("Settings")
    model_option = st.sidebar.selectbox("Pick model", ["Logistic Regression", "Random Forest", "XGBoost"])
    
    tune_hyperparams = False
    if model_option != "Logistic Regression":
        tune_hyperparams = st.sidebar.checkbox("Use Hyperparameter Tuning", value=False)


    if st.sidebar.button("Train", type="primary"):
        
        st.subheader(f"Training: {model_option}")
        with st.spinner("Training in progress..."):
            
            if model_option == "Logistic Regression":
                model = tr.train_logistic_regression(X_train, y_train)
            elif model_option == "Random Forest":
                model = tr.train_random_forest(X_train, y_train, tune=tune_hyperparams)
            elif model_option == "XGBoost":
                model = tr.train_xgboost(X_train, y_train, tune=tune_hyperparams)
            
            acc, report, cm, precision, recall, f1 = tr.evaluate_model(model, X_val, y_val)
        
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall", f"{recall:.2%}")
            col4.metric("F1 Score", f"{f1:.2%}")
            st.text("Classification report:")
            st.code(report)
            
            if tune_hyperparams and hasattr(model, 'best_params_'):
                st.success(f"Best hiperparameters: {model.best_params_}")

            st.write("Confusion matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)

            

else:
    st.warning("Data is not available for training.")