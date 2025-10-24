import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Data division
def split_data(df_processed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the preprocessed DataFrame into training and testing sets

    df_processed (pd.DataFrame): The clean DataFrame (after initial_preprocess)
    """
    y: pd.Series = df_processed['Class']
    X: pd.DataFrame = df_processed.drop('Class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def create_smote_pipeline() -> Pipeline:
    """
    Creates a imbalanced-learn (imblearn) pipeline that aplies SMOTE and trains an LogisticRegression classifier
    """
    pipeline: Pipeline = Pipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('model', LogisticRegression(random_state=42, max_iter=1000))
    ])
    return pipeline

def create_smote_rf_pipeline():
    """
    Creates a imbalanced-learn (imblearn) pipeline that aplies SMOTE and trains an RandomForest classifier
    """
    pipeline = Pipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('random_forest', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return pipeline