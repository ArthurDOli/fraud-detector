import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys

### Load the dataset
def read_archive(file_path):
    """
    Load the dataset and stop the code execution when an error occurs.
    """
    try:
        df = pd.read_csv(f'{file_path}')
        return df
    except FileNotFoundError:
        print("Error: File not found")
        sys.exit(1)
    except Exception as e:
        print(f"An error ocurred when loading the file {e}")
        sys.exit(1)


### Preprocessing
def initial_preprocess(df):
    """
    Performs the initial preprocessing and removes the 'Time' column and scales the 'Amount' column.
    """
    if df is None:
        return None

    df_processed = df.copy()

    # Removes 'Time' column
    if 'Time' in df_processed:
        df_processed = df_processed.drop('Time', axis=1)
    else:
        print('Column "Time" not found')

    # Scales 'Amount' column
    if 'Amount' in df_processed:
        scaler = StandardScaler()
        df_processed['Amount'] = scaler.fit_transform(df_processed[['Amount']])
    else:
        print('Column "Amount" not found')

    # Scales 'V' columns
    v_features = [f'V{i}' for i in range(1, 29)]
    if all(feature in df_processed.columns for feature in v_features):
        df_processed[v_features] = scaler.fit_transform(df_processed[[v_features]])
    else:
        print('Some columns were not found')

    return df_processed