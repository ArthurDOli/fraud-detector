from data_processing import load_data, initial_preprocess
# from model_training import 
# from evaluation import 
import sys
from model_training import split_data

file = 'data/creditcard.csv'

df = load_data(file)

# Verificação do DataFrame
if df is not None:
    df_new = initial_preprocess(df)
    if df_new is not None:
        X_train, X_test, y_train, y_test = split_data(df_new)
        print(X_train.shape)
    else:
        print('An error occurred during the preprocessing')
        sys.exit(1)
else:
    print('Fail loading the data')
    sys.exit(1)

