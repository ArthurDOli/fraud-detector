import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df_processed: pd.DataFrame):
    y: pd.DataFrame = df_processed['Class']
    X: pd.DataFrame = df_processed.drop('Class', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test