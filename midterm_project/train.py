#!/usr/bin/env python
# coding: utf-8

# ## Midterm Alex Khvatov

import pickle

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path

pd.options.mode.copy_on_write = True

#Parameters

output_file = "model.bin"

columns = ['age', 'sex', 'cp', 'trestbps', 'chol','fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

path_to_data_file = Path.resolve(Path("./data/processed.cleveland.data"))
p_cleveland_df = pd.read_csv(path_to_data_file, header=None, names= columns )

string_columns = list(p_cleveland_df.dtypes[p_cleveland_df.dtypes == 'object'].index)
for c in string_columns:
    p_cleveland_df[c] = p_cleveland_df[c].str.strip()


p_cleveland_df = p_cleveland_df[p_cleveland_df['thal']!='?']

#Missing data for 'ca' column (number of major vessels (0-3) colored by flourosopy) we are going to set to 0
p_cleveland_df.loc[p_cleveland_df['ca']=='?', 'ca'] = 0.0

int_columns = ['age', 'sex', 'cp', 'trestbps', 'chol','fbs', 'restecg', 'thalach', 'exang', 'slope', 'ca', 'thal', 'num']
float_columns = ['oldpeak']


for c in int_columns:
    p_cleveland_df[c] = pd.to_numeric(p_cleveland_df[c], errors='coerce')
    p_cleveland_df[c] = p_cleveland_df[c].astype(int)

#At this point we have 'clean' dataset.
#Because 'num' can be 0 - for healthy and other numbers for having presence of a heart disease we can change all the numbers to 1
p_cleveland_df.loc[p_cleveland_df['num']!=0, 'num'] = 1


categorical_columns = ['sex', 'cp', 'fbs', 'restecg',  'exang',  'slope', 'ca', 'thal']
numeric_columns = ['age', 'trestbps', 'chol', 'thalach','oldpeak']

for c in categorical_columns:
    p_cleveland_df[c] = p_cleveland_df[c].astype(str)

df_full_train, df_test = train_test_split(p_cleveland_df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size = 0.25, random_state=1)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train = df_train.num.values
y_val = df_val.num.values
y_test = df_test.num.values

del df_train['num']
del df_val['num']
del df_test['num']

train_dicts=df_train[categorical_columns + numeric_columns].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical_columns + numeric_columns].to_dict(orient='records')
X_val = dv.transform(val_dicts)


def train(df_train, y_train, C=1.0):
    """Trains Logistic Regression model

    Args:
        df_train (_type_): training dataset
        y_train (_type_): training target values
        C (float, optional): Inverse of regularization strength. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    dicts = df_train[categorical_columns + numeric_columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train=dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train, y_train)
    return dv, model

def predict(df, dv, model):
    dicts = df[categorical_columns + numeric_columns].to_dict(orient = "records")
    X=dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred

dv, logistic_regression_model = train(df_full_train, df_full_train.num.values, C=1.0)

y_pred = predict(df_test, dv, logistic_regression_model)

auc = roc_auc_score(y_test, y_pred)

print(f"auc of the final model={auc:.3f}")

# ### Save the model


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, logistic_regression_model), f_out)
    
print(f"Model saved to {output_file}")
