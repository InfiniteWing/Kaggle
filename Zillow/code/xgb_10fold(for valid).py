#
# This script is inspired by this discussion:
# https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# Author: InfiniteWing
# Ver 4. updated the dataset
# LB: 0.06450
#

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

properties = pd.read_csv('../input/properties_2017.csv')
train_2016 = pd.read_csv("../input/train_2016_v2.csv")
train_2017 = pd.read_csv("../input/train_train_2017.csv")
valid2017 = pd.read_csv('../input/train_valid_2017.csv')
train = pd.concat([train_2016, train_2017], ignore_index=True)

for c in properties.columns:
    if properties[c].dtype == 'object':
        properties[c]=properties[c].fillna(-1)
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))

train_df = train.merge(properties, how='left', on='parcelid')
valid2017_df = valid2017.merge(properties, how='left', on='parcelid')
# drop out ouliers
train_df = train_df[ train_df.logerror > -0.4 ]
train_df = train_df[ train_df.logerror < 0.4 ]

x_train = np.array(train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1))
y_train = np.array(train_df["logerror"].values.astype(np.float32))
x_valid = np.array(valid2017_df.drop(['parcelid', 'logerror','transactiondate'], axis=1))
y_valid = np.array(valid2017_df["logerror"].values.astype(np.float32))
y_mean = np.mean(y_train)



K = 5
kf = KFold(n_splits = K, random_state = 3228, shuffle = True)
fold = 1
y_preds = None
valid_scores = 0.0
for train_index, test_index in kf.split(x_train):
    print("Start fold", fold)
    train_X, valid_X = x_train[train_index], x_train[test_index]
    train_y, valid_y = y_train[train_index], y_train[test_index]
    
    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(x_valid)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    # xgboost params
    xgb_params = {
        'eta': 0.02,
        'max_depth': 6,
        'subsample': 0.85,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'seed': 3228,
        'silent': 1
    }
    model = xgb.train(xgb_params, d_train, 360, watchlist, early_stopping_rounds=20, verbose_eval=30)
    pred = model.predict(d_test)
    
    valid_score = mean_absolute_error(y_valid, pred)
    print('Valid MAE =', valid_score)
    valid_scores += valid_score
    if(fold == 1):
        y_preds = pred
    else:
        y_preds += pred
    fold += 1
    
valid_scores /= K
y_preds /= K
print('Average valid MAE =', valid_scores)

valid_score = mean_absolute_error(y_valid, y_preds)

print('Blending valid MAE =', valid_score)

output = pd.DataFrame({'valid_pred': y_preds})
output.to_csv('valid_xgb.csv', index=False, float_format = '%.5f')

