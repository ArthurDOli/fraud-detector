import sys
import os

project = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project)

import pytest
import pandas as pd
from src.data_processing import load_data, initial_preprocess


def test_load_data_success():
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
