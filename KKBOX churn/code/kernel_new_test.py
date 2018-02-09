import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn

test = pd.read_csv('../input/sample_submission_v2.csv')
train = pd.read_csv('train_pre_201703.csv')

test = pd.merge(test, train, how='left', on='msno')

train = pd.read_csv('train_pre_201703.csv')
members = pd.read_csv('../input/members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = []; print('members merge...') 
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(-1)
test = test.fillna(-1)

cols = [c for c in train.columns if c not in ['is_churn','msno']]
print(cols)
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

fold = 1
for i in range(fold):
    
    model = xgb.Booster()
    model.load_model('xgb_model_{}.model'.format(i))
    
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]))
    else:
        pred = model.predict(xgb.DMatrix(test[cols]))
    
pred /= fold
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('sub.csv', index=False)
#test[['msno','is_churn']].to_csv('submission.csv.gz', index=False, compression='gzip')
