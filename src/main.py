from data_processing import load_data, initial_preprocess
# from model_training import 
# from evaluation import 
import sys

file = 'data/creditcard.csv'

df = load_data(file)

# Verificação do DataFrame
if df is not None:
    df_new = initial_preprocess(df)
    if df_new is not None:
        print(df_new.describe())
    else:
        print('An error occurred during the preprocessing')
        sys.exit(1)
else:
    print('Fail loading the data')
    sys.exit(1)

