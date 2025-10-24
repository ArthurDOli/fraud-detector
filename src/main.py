from data_processing import load_data, initial_preprocess
# from model_training import 
from evaluation import get_cv_scores
import sys
from model_training import split_data
from sklearn.linear_model import LogisticRegression

file = 'data/creditcard.csv'

df = load_data(file)

# Verificação do DataFrame
if df is not None:
    df_new = initial_preprocess(df)
    if df_new is not None:
        X_train, X_test, y_train, y_test = split_data(df_new)
        baseline_model = LogisticRegression()
        print(get_cv_scores(baseline_model, X_train, y_train))
    else:
        print('An error occurred during the preprocessing')
        sys.exit(1)
else:
    print('Fail loading the data')
    sys.exit(1)

