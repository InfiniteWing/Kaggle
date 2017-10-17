import numpy as np
import pandas as pd
import xgboost as xgb

train2017 = pd.read_csv('../input/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
train2017["days"] = train2017["transactiondate"].dt.month * 31 + train2017["transactiondate"].dt.day
train2017_train = train2017[ train2017.days <  231]
train2017_valid = train2017[ train2017.days >=  231]
train2017_train = train2017_train.drop(['days'], axis = 1)
train2017_valid = train2017_valid.drop(['days'], axis = 1)

train2017_train.to_csv('../input/train_train_2017.csv',index=False)
train2017_valid.to_csv('../input/train_valid_2017.csv',index=False)