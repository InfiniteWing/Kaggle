import gc; gc.enable()
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('xgb_valid_201703.csv')
train['is_churn_xgb'] = pd.read_csv('xgb_valid_201703.csv')['is_churn']
train['is_churn_lgb'] = pd.read_csv('lgb_valid_201703.csv')['is_churn']
train['is_churn_cat'] = pd.read_csv('cat_valid_201703.csv')['is_churn']
train['is_churn_cat'] = 1 - train['is_churn_cat']
train['is_churn'] = pd.read_csv('train_pre_v4_201703.csv')['is_churn']

cols = [c for c in train.columns if c not in ['is_churn','msno']]
print(cols)
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)
pred_all = 0.0
fold = 1
for i in range(fold):
    params = {
        'eta': 0.07,
        'max_depth': 7,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': 3228,
        'silent': True,
        'tree_method': 'exact'
    }
    lgb_params = {
        'learning_rate': 0.05,
        'application': 'binary',
        'max_depth': 7,
        'num_leaves': 256,
        'verbosity': -1,
        'metric': 'binary_logloss'
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.2, random_state=i)
    '''
    x1 = train[cols]
    y1 = train['is_churn']
    x2 = valid[cols]
    y2 = valid['is_churn']
    '''
    # xgb
    model = xgb.Booster()
    model.load_model('xgb_stacking_model_0_limit_143.model')
    xgb_pred = model.predict(xgb.DMatrix(train[cols]), ntree_limit=143)
    print('stacking xgb log loss = {}'.format(log_loss(train['is_churn'],xgb_pred)))
    
    # lgbm
    model = lgb.Booster(model_file='lgb_stacking_model_{}.model'.format(i))
    lgb_pred = model.predict(train[cols])
    print('stacking lgb log loss = {}'.format(log_loss(train['is_churn'],lgb_pred)))
    print('stacking lgb log loss = {}'.format(log_loss(train['is_churn'],train['is_churn_cat'])))
    
    