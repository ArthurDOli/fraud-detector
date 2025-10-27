import sys
import os

project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project)

import pytest
import pandas as pd
from src.evaluation import get_cv_scores, get_final_metrics
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List

def test_get_cv_scores():
    """
    Test of get_cv_scores uses cross_validation correctly
    """
    fake_data: pd.DataFrame = pd.DataFrame(data={'Class': np.concatenate((np.zeros(100), np.ones(10))), 'Feature': np.random.rand(110)})
    X_fake: pd.DataFrame = fake_data[['Feature']]
    y_fake: pd.Series = fake_data['Class']
    logistic_regression = LogisticRegression()
    scores = get_cv_scores(model=logistic_regression, X_train=X_fake, y_train=y_fake)
    assert isinstance(scores, Dict)
    assert 'mean_precision' in scores.keys()

def test_get_final_metrics():
    """
    Test if the get_final_metrics function really works
    """
    logistic_regression: LogisticRegression = LogisticRegression()
    X_train_fake: List[List[int]] = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    X_test_fake: List[List[int]] = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    y_train_fake: List[int] = [0, 1, 0, 1]
    y_test_fake: List[int] = [0, 1, 0, 1]
    results: LogisticRegression = logistic_regression.fit(X_train_fake, y_train_fake)
    metrics: Dict[str, float] = get_final_metrics(results, X_test_fake, y_test_fake)
    assert isinstance(metrics, Dict)