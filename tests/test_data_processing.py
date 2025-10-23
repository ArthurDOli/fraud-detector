import sys
import os

project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project)

import pytest
import pandas as pd
from src.data_processing import load_data, initial_preprocess


def test_load_data_success():
    """
    Tests if the load_data returned a DataFrame
    """
    data_path = 'data/creditcard.csv'
    df = load_data(data_path)
    assert df is not None
    assert isinstance(df, pd.DataFrame)

def test_load_data_file_not_found():
    """
    Tests if load_data returns None when the archive doesn't exists.
    """
    df = load_data('a')
    assert df is None

def test_initial_preprocess_time_column_removed():
    """
    Tests if the initial_preprocess has removed the 'Time' column
    """
    fake_data: dict = {'Time': [1, 2], 'Amount': [10, 20], 'Class': [0, 1]}
    fake_data_df = pd.DataFrame(fake_data)
    process: pd.DataFrame = initial_preprocess(fake_data_df)
    assert 'Time' not in process.columns
    assert 'Amount' in process.columns

def test_initial_preprocess_scales_amount():
    """
    Tests if initial_preprocess has scaled 'Amount' column 
    """
    fake_data: dict = {'Time': [1, 2, 3], 'Amount': [10, 20, 30], 'Class': [0, 0, 1]}
    fake_data_df = pd.DataFrame(fake_data)
    process: pd.DataFrame = initial_preprocess(fake_data_df)
    assert process['Amount'].mean() == pytest.approx(0.0)
    assert process['Amount'].std(ddof=0) == pytest.approx(1.0)