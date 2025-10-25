from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
from imblearn.pipeline import Pipeline

def get_cv_scores(model: Any, X_train: pd.DataFrame, y_train:pd.Series) -> Dict[str, float]:
    """
    Calculates the scores of the cross validation stratified (StratifiedKFold) for each data in a model.
    """
    stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics: list[str] = ['precision', 'recall', 'f1', 'average_precision']
    scoring = cross_validate(model, X_train, y_train, cv=stratified, scoring=metrics)
    return {f'mean_{metric}': scoring[f'test_{metric}'].mean() for metric in metrics}

def get_final_metrics(model_trained: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates and prints the final classification metrics for the tests set
    """
    y_pred: np.ndarray = model_trained.predict(X_test)
    y_prob: np.ndarray = model_trained.predict_proba(X_test)[:, 1]
    p_score = precision_score(y_test, y_pred)
    r_score = recall_score(y_test, y_pred)
    f_score = f1_score(y_test, y_pred)
    average_score = average_precision_score(y_test, y_prob)
    roc_score = roc_auc_score(y_test, y_prob)
    print(confusion_matrix(y_test, y_pred))
    return {'precision': p_score, 'recall': r_score, 'f1': f_score, 'average_precision_score': average_score, 'roc_auc': roc_score}