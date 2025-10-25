import sys
import os

project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project)

import pytest
import pandas as pd
from src.model_training import split_data, create_smote_pipeline, create_smote_rf_pipeline
import numpy as np
from imblearn.pipeline import Pipeline
from typing import Tuple

def test_split_data_returns_four_elements():
    """
    Tests if the 'split_data' returns a tuple with four elements (X_train, X_test, y_train, y_test)
    """
    fake_data: dict = {'Class': [0, 1, 0, 1], 'feature': [1, 2, 3, 4]}
    fake_data_df: pd.DataFrame = pd.DataFrame.from_dict(fake_data)
    result: Tuple = split_data(fake_data_df)
    print(result)
    assert len(result) == 4

def test_split_data_stratification():
    """
    Tests if 'stratify=y' maintains the proportion of the minority class in the training and tests sets
    """
    fake_data: dict = {'Class': np.concatenate((np.zeros(100), np.ones(10))), 'Feature': np.random.rand(110)}
    _, _, y_train, y_test = split_data(pd.DataFrame(fake_data))
    prop: float = 10/110

    proportion_train: pd.Series = y_train.value_counts(normalize=True)
    prop_train: float = proportion_train[1]

    proportion_test: pd.Series = y_test.value_counts(normalize=True)
    prop_test: float = proportion_test[1]

    assert prop_train == pytest.approx(prop)
    assert prop_test == pytest.approx(prop)

def test_create_somte_rf_pipeline_returns_pipeline():
    """
    Tests if 'create_smote_rf_pipeline' function really returned a Pipeline
    """
    pipeline: Pipeline = create_smote_rf_pipeline()
    assert isinstance(pipeline, Pipeline)

def test_create_smote_rf_pipeline_has_correct_steps():
    """
    Verify if the 'create_smote_rf_pipeline' pipeline contains steps with the correct names ('smote', 'random_forest')
    """
    pipeline: Pipeline = create_smote_rf_pipeline()
    assert 'smote' in pipeline.named_steps
    assert 'random_forest' in pipeline.named_steps