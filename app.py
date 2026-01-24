import streamlit as st
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


try:
    import preprocessing as pp
    import training as tr
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(page_title="Titanic App", layout="centered")

@st.cache_data
def get_data():
    """
    Load raw data - preprocessing will happen inside Pipeline to prevent leakage
    """
    try:
        # Load raw data only
        data = pp.load_data('train.csv')
        
        X = data.drop('Survived', axis=1)
        y = data['Survived']
        
        return X, y
        
    except FileNotFoundError:
        st.error("Source data file not found.")
        return None, None
    except Exception as e:
        st.error(f"Error while loading data: {e}")
        return None, None

# Main App
st.title("Titanic Training Platform")

X, y = get_data()

if X is not None:
    st.sidebar.header("Settings")
    # Select model
    model_option = st.sidebar.radio("Pick model", ["Logistic Regression", "Random Forest", "XGBoost"])
    
    # Choose between GridSearch and Manual parameters
    mode = st.sidebar.radio("Parameter Mode", ["Manual Parameters", "GridSearchCV"])
    
    param_grid = None
    manual_params = {}
    
    if mode == "GridSearchCV":
        st.sidebar.subheader("GridSearch Parameters")
        
        # Metric selection
        scoring_metric = st.sidebar.radio(
            "Scoring Metric",
            ['accuracy', 'precision', 'recall', 'f1'],
            index=0
        )
        
        if model_option == "Logistic Regression":
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['lbfgs', 'saga', 'liblinear'],
                'penalty': ['l1', 'l2']
            }
            st.sidebar.markdown("• **C:** 0.01, 0.1, 1.0, 10.0, 100.0")
            st.sidebar.markdown("• **Solver:** lbfgs, saga, liblinear")
            st.sidebar.markdown("• **Penalty:** l1, l2")
        
        elif model_option == "Random Forest":
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10]
            }
            st.sidebar.markdown("• **N Estimators:** 50, 100, 200")
            st.sidebar.markdown("• **Max Depth:** 5, 10, 15, None")
            st.sidebar.markdown("• **Min Samples Split:** 2, 5, 10")
        
        elif model_option == "XGBoost":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1]
            }
            st.sidebar.markdown("• **N Estimators:** 100, 200")
            st.sidebar.markdown("• **Max Depth:** 3, 5, 7")
            st.sidebar.markdown("• **Learning Rate:** 0.05, 0.1")
    
    elif mode == "Manual Parameters":
        st.sidebar.subheader("Manual Parameters")

        scoring_metric = None  # Not used in manual mode
        if model_option == "Logistic Regression":
            manual_params['C'] = st.sidebar.select_slider("C (Regularization)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            manual_params['max_iter'] = st.sidebar.slider("Max Iterations", 100, 2000, 1000, 100)
            manual_params['penalty'] = st.sidebar.radio("Penalty", ['l1', 'l2'], index=0)
            if manual_params['penalty'] == 'l1':
                manual_params['solver'] = 'saga'
            else:
                manual_params['solver'] = st.sidebar.radio("Solver", ['lbfgs', 'saga'], index=0)
        
        elif model_option == "Random Forest":
            manual_params['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 500, 100, 10)
            manual_params['max_depth'] = st.sidebar.slider("Max Depth (0=None)", 0, 20, 10, 1)
            if manual_params['max_depth'] == 0:
                manual_params['max_depth'] = None
            manual_params['min_samples_split'] = st.sidebar.slider("Min Samples Split", 2, 10, 2, 1)
        
        elif model_option == "XGBoost":
            manual_params['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
            manual_params['max_depth'] = st.sidebar.slider("Max Depth", 3, 15, 5, 1)
            manual_params['learning_rate'] = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)

    if st.sidebar.button("Train", type="primary"):
        
        st.subheader(f"Training: {model_option}")
        
        # K-Fold CV settings
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Create preprocessing pipeline 
        preprocessing_pipeline = pp.create_preprocessing_pipeline()
        
        # Display mode info
        if mode == "GridSearchCV":
            # Calculate total combinations
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)
            st.info(f"Using best parameters (optimized for {scoring_metric}) on full dataset")
            
        else:
            st.info(f"Using manually selected parameters on full dataset")
        
        with st.spinner("Training in progress..."):
            
            if model_option == "Logistic Regression":
                model = tr.train_logistic_regression(preprocessing_pipeline, X, y, param_grid=param_grid, manual_params=manual_params if mode == "Manual Parameters" else None, scoring=scoring_metric if mode == "GridSearchCV" else None)
            elif model_option == "Random Forest":
                model = tr.train_random_forest(preprocessing_pipeline, X, y, param_grid=param_grid, manual_params=manual_params if mode == "Manual Parameters" else None, scoring=scoring_metric if mode == "GridSearchCV" else None)
            elif model_option == "XGBoost":
                model = tr.train_xgboost(preprocessing_pipeline, X, y, param_grid=param_grid, manual_params=manual_params if mode == "Manual Parameters" else None, scoring=scoring_metric if mode == "GridSearchCV" else None)
            
            
            # Perform cross-validation evaluation
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring_metrics, return_train_score=False)
            
            # Calculate mean scores
            acc = np.mean(cv_results['test_accuracy'])
            precision = np.mean(cv_results['test_precision'])
            recall = np.mean(cv_results['test_recall'])
            f1 = np.mean(cv_results['test_f1'])
            
            # Train final model on full data for confusion matrix
            if mode == "GridSearchCV":
                final_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model 
            else:
                final_model = model
            
            # Get predictions for confusion matrix
            y_pred = cross_val_predict(final_model, X, y, cv=cv)
            
            # Generate classification report
            report = classification_report(y, y_pred, output_dict=True)
            cm = confusion_matrix(y, y_pred)
            
            # Prepare DataFrames for display
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.round(3)
            
            cm_df = pd.DataFrame(
                cm,
                index=['Actual: Not Survived', 'Actual: Survived'],
                columns=['Predicted: Not Survived', 'Predicted: Survived']
            )

            # Store all results in session state
            st.session_state['trained_model'] = final_model
            st.session_state['model_name'] = model_option
            st.session_state['cv_results'] = {
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'report_df': report_df,
                'cm_df': cm_df,
                'best_params': model.best_params_ if (mode == "GridSearchCV" and hasattr(model, 'best_params_')) else None,
                'mode': mode,
                'n_splits': n_splits
            }
    
    # Display results if they exist in session state (persists after rerun)
    if 'cv_results' in st.session_state:
        results = st.session_state['cv_results']
        
        st.subheader("Cross-Validation Results")
        st.caption(f"Scores averaged across {results['n_splits']} folds using full dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{results['accuracy']:.2%}")
        col2.metric("Precision", f"{results['precision']:.2%}")
        col3.metric("Recall", f"{results['recall']:.2%}")
        col4.metric("F1 Score", f"{results['f1']:.2%}")
        
        st.subheader("Classification Report")
        st.caption("Detailed per-class metrics from cross-validation predictions")
        st.dataframe(results['report_df'], use_container_width=True)

        st.subheader("Confusion Matrix")
        st.caption("Aggregated predictions from all folds")
        st.dataframe(results['cm_df'], use_container_width=True)

        if results['best_params']:
            st.subheader("Best Parameters")
            st.caption(f"**Best parameters found:** {results['best_params']}")
            

    # Prediction section for test.csv
    if 'trained_model' in st.session_state:
        st.subheader("Generate Predictions")
        st.caption(f"Use the trained **{st.session_state['model_name']}** model to predict on test set and download submission.csv for Kaggle Competition")
        
        try:
            # Load raw test data
            test = pp.load_data('test.csv')
            
            # Save PassengerId before any transformations
            passenger_ids = test['PassengerId'].copy()
            
            # Use the same trained pipeline (already fitted on training data)
            predictions = st.session_state['trained_model'].predict(test)
            
            # Create submission DataFrame
            submission = pd.DataFrame({
                'PassengerId': passenger_ids,
                'Survived': predictions
            })
            
            # Save to CSV
            submission.to_csv('submission.csv', index=False)
            
            # Convert to CSV for download
            csv = submission.to_csv(index=False)
            
            # Download button that generates and downloads in one click
            st.download_button(
                label="Download submission.csv",
                data=csv,
                file_name="submission.csv",
                mime="text/csv",
                type="secondary" 
            )
                
        except Exception as e:
            st.error(f"Error preparing predictions: {str(e)}")
            
else:
    st.warning("Data is not available for training.")