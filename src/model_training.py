import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from typing import Tuple
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
from typing import Dict, Any

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

def create_smote_rf_pipeline() -> Pipeline:
    """
    Creates a imbalanced-learn (imblearn) pipeline that aplies SMOTE and trains an RandomForest classifier
    """
    pipeline = Pipeline(steps=[
        ('smote', SMOTE(random_state=42)),
        ('random_forest', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    return pipeline

# WARNING: Running the RandomizedSearchCV within this function is computationally very expensive and may
# take a significant amount of time depending on the hardware, the dataset size, the number of iterations,
# and the cross validations folds.
# def tune_rf_pipeline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
#     """
#     Note: This function is computationally expensive due to RandomizedSearchCV.
    
#     Executes RandomizedSearchCV in a SMOTE + RandomForestClassifier Pipeline.

#     Aims for the best hyperparameters for RandomForestClassifier in the pipeline, optimizing the metric
#     'average_precision' stratified cross validation.
#     """
#     pipeline = Pipeline(steps=[
#         ('smote', SMOTE(random_state=42)),
#         ('random_forest', RandomForestClassifier(random_state=42, n_jobs=-1))
#     ])
#     params: Dict[str, Any] = {'random_forest__n_estimators': randint(100, 500), 
#                                                     'random_forest__max_depth': randint(10, 50), 
#                                                     'random_forest__min_samples_split': randint(2, 11),
#                                                     'random_forest__min_samples_leaf': randint(1, 6), 
#                                                     'random_forest__class_weight': ['balanced', None]}
#     stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     randomized_seach = RandomizedSearchCV(estimator=pipeline, param_distributions=params, n_iter=20, 
                                          
#                                           # 'scoring' defines the metric that will be optimized. For unbalanced data like
#                                           # fraud detection, 'accuracy' is useless.
#                                           # 'average_precision' focus on the performance of the minority class (fraud),
#                                           # seeking a good overall balance.
#                                           # 'f1' algo would be a good option if the objective was find the best balance in a specific threshold.
#                                           scoring='average_precision',
                                          
#                                           cv=stratified,
#                                           random_state=42,
#                                           n_jobs=-1,
#                                           verbose=2)
#     search = randomized_seach.fit(X_train, y_train)
#     return search.best_estimator_