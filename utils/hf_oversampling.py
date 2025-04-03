import pandas as pd
import numpy as np
import logging
import json
import smote_variants as sv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.metrics import classification_report_imbalanced 

# Setup logging
logging.basicConfig(
    filename='../logs/oversampling_experiments_v2.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def preprocess_data(df, features, target):
    """
    Preprocess the data:
     - One-hot encode the features to get numeric columns.
     - Label encode the target if necessary.
     - Return processed X and y.
    """
    # One-hot encode the predictor columns
    X = pd.get_dummies(df[features], drop_first=True)
    
    # Label encode the target column if it's not numeric
    if df[target].dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(df[target])
    else:
        y = df[target].values

    return X, y

def evaluate_classifiers(X_train, y_train, X_test, y_test, method_name):
    """
    Given training and testing data, trains logistic regression and random forest,
    evaluates accuracy and F1-score, and logs the results.
    """
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr, average='micro')
    report_lr = classification_report(y_test, y_pred_lr)
    logger.info(f"{method_name} - Logistic Regression: Accuracy: {acc_lr:.4f}, Micro-average F1: {f1_lr:.4f}")
    logger.info(f"Classification Report:\n{report_lr}")
    results['Logistic Regression'] = {'accuracy': acc_lr, 'f1': f1_lr, 'report': report_lr}
    print(f"{method_name} - Logistic Regression: Accuracy: {acc_lr:.4f}, Micro-average F1: {f1_lr:.4f}")
    
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='micro')
    report_rf = classification_report(y_test, y_pred_rf)
    logger.info(f"{method_name} - Random Forest: Accuracy: {acc_rf:.4f}, Micro-average F1: {f1_rf:.4f}")
    logger.info(f"Classification Report:\n{report_rf}")
    results['Random Forest'] = {'accuracy': acc_rf, 'f1': f1_rf, 'report': report_rf}
    print(f"{method_name} - Random Forest: Accuracy: {acc_rf:.4f}, Micro-average F1: {f1_rf:.4f}")
    
    return results

def evaluate_smote_variants(df, features, target, oversample_methods):
    """
    For each oversampling method from smote_variants, perform oversampling on training data,
    then evaluate logistic regression and random forest on the test set.
    """
    # Preprocess data: convert categorical features to numeric (one-hot encode) and label encode target.
    X, y = preprocess_data(df, features, target)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    overall_results = {}
    
    # Loop through each oversampling method
    for method_name, method_func in oversample_methods.items():
        print(f"Evaluating oversampling method: {method_name}")
        logger.info(f"Evaluating oversampling method: {method_name}")
        
        # The smote_variants functions expect numpy arrays.
        # X_train_np = X_train.values
        # y_train_np = y_train
        
        try:
            # Apply oversampling
            oversampler = sv.MulticlassOversampling(oversampler=method_name, oversampler_params={'random_state': 5})
            X_res, y_res = oversampler.sample(X_train, y_train)
            logger.info(f"{method_name}: Resampled distribution: {np.unique(y_res, return_counts=True)}")
        except Exception as e:
            logger.error(f"Error with {method_name}: {e}")
            continue
        
        # Convert resampled X to DataFrame with same columns as X_train
        X_res_df = pd.DataFrame(X_res, columns=X_train.columns)
        
        # Evaluate classifiers on the resampled data
        results = evaluate_classifiers(X_res_df, y_res, X_test, y_test, method_name)
        overall_results[method_name] = results
    
    return overall_results

if __name__ == '__main__':
    # Load your dataset (ensure that the dataset is preprocessed such that features and target are in the df)
    df = pd.read_csv('../data/hf_models_withmodelcard_nov2024.csv')
    
    features = ['task_group', 'author_category', 'language_category', 'location']
    target = 'downloads_category'
    
    # Define oversampling methods from smote_variants to try:
    oversample_methods = {
        'SMOTE': sv.SMOTE(),
        'distance_SMOTE': sv.distance_SMOTE(),
        'Borderline_SMOTE1': sv.Borderline_SMOTE1(),
        'ADASYN': sv.ADASYN(),
        'MWMOTE': sv.MWMOTE(),
        'kmeans_SMOTE': sv.kmeans_SMOTE(),
        'Supervised_SMOTE': sv.Supervised_SMOTE(),
        'SMOTE_TomekLinks': sv.SMOTE_TomekLinks(),
        'SMOTE_Cosine': sv.SMOTE_Cosine(),
        'polynom_fit_SMOTE_poly': sv.polynom_fit_SMOTE_poly()
    }
    
    results = evaluate_smote_variants(df, features, target, oversample_methods)
    
    # Optionally print or save the results
    print(json.dumps(results, indent=2))
    logger.info("Final results:")
    logger.info(json.dumps(results, indent=2))
