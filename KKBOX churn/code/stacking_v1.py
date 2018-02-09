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

train = pd.read_csv('valid_stacking_201702.csv')

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
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 200,  watchlist, feval=xgb_score, maximize=False, verbose_eval=10, early_stopping_rounds=10)
    #model = xgb.Booster()
    #model.load_model('xgb_model_0_limit_149.model')
    model.save_model('xgb_stacking_model_{}_limit_{}.model'.format(i, model.best_ntree_limit))
    #print(model.best_ntree_limit)
    #xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)

    '''
    # RandomForest
    rf = RandomForestClassifier(n_estimators=300, n_jobs=8,random_state=3228)
    rf.fit(x1, y1)
    pred = rf.predict_proba(x2)
    pred1 = 1 - pred
    print('log loss = {}'.format(log_loss(y2,pred)))
    print('log loss = {}'.format(log_loss(y2,pred1)))
    '''
    
    # lgbm
    d_train = lgb.Dataset(x1, label=y1)
    d_valid = lgb.Dataset(x2, label=y2)
    watchlist = [d_train, d_valid]

    model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=10) 
    model.save_model('lgb_stacking_model_{}.model'.format(i))
    
    
    # catboost
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.12,
        depth=7, l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=3228)
    model = model.fit(x1, y1,eval_set=(x2,y2))
    model.save_model('cat_stacking_model_{}.model'.format(i))
    