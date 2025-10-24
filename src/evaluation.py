from sklearn.model_selection import cross_validate, StratifiedKFold
import pandas as pd
import numpy as np
from typing import Dict, Any

def get_cv_scores(model: Any, X_train: pd.DataFrame, y_train:pd.Series) -> Dict[str, float]:
    """
    Calculate the scores of the cross validation stratified (StratifiedKFold) for each data in a model.
    """
    stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics: list[str] = ['precision', 'recall', 'f1', 'average_precision']
    scoring = cross_validate(model, X_train, y_train, cv=stratified, scoring=metrics)
    return {f'mean_{metric}': scoring[f'test_{metric}'].mean() for metric in metrics}