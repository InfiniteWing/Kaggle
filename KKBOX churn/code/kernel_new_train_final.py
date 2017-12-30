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
import math
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('train_pre_2017_all_v2.csv')
test = pd.read_csv('../input/sample_submission_v2.csv')
test_features = pd.read_csv('train_pre_v2_201704.csv')
test_features = test_features.drop(['is_churn'],axis=1)
test = pd.merge(test, test_features, how='left', on='msno')

members = pd.read_csv('../input/members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno') 
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)
train = train.fillna(-1)
test = test.fillna(-1)

#cols = [c for c in train.columns if c not in ['is_churn','msno']]
cols = [c for c in train.columns if c not in ['is_churn','msno','transaction_date','expire_date']]
print(cols)
def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)
xgb_preds = 0.0
lgb_preds = 0.0
cat_preds = 0.0
fold = 3
for i in range(fold):
    print('start fold {}'.format(i+1))
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
        
    # catboost
    #print('cat training')
    '''
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.12,
        depth=7, l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=3228)
    #model.load_model('cat_model_{}.model'.format(i))
    model = model.fit(x1, y1,eval_set=(x2,y2),logging_level='Silent')
    cat_pred = model.predict_proba(test[cols])[:,1]
    
    cat_valid = model.predict_proba(x2)[:,1]
    print('cat valid log loss = {}'.format(log_loss(y2,cat_valid)))
    model.save_model('cat_model_v2_{}.model'.format(i))
    
    
    # xgb
    #print('xgb training')
    #ntree_limit = [199,193,190,185,199]
    #model = xgb.Booster()
    #model.load_model('xgb_model_{}_limit_{}.model'.format(i,ntree_limit[i]))
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 200,  watchlist, feval=xgb_score, maximize=False, verbose_eval=100, early_stopping_rounds=10)
    model.save_model('xgb_model_v2_{}_limit_{}.model'.format(i, model.best_ntree_limit))
    xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    #xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=ntree_limit[i])
    xgb_valid = model.predict(xgb.DMatrix(x2))
    print('xgb valid log loss = {}'.format(log_loss(y2,xgb_valid)))
    '''
    # lgbm
    #print('lgb training')
    d_train = lgb.Dataset(x1, label=y1)
    d_valid = lgb.Dataset(x2, label=y2)
    watchlist = [d_train, d_valid]

    #model = lgb.train(lgb_params, train_set=d_train, num_boost_round=240, valid_sets=watchlist, early_stopping_rounds=50, verbose_eval=100) 
    model = lgb.Booster(model_file='lgb_model_v2_{}.model'.format(i))
    ax = lgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig('feature_importance_{}.png'.format(i))
    break
    lgb_pred = model.predict(test[cols])
    model.save_model('lgb_model_v2_{}.model'.format(i))
    lgb_valid = model.predict(x2)
    print('lgb valid log loss = {}'.format(log_loss(y2,lgb_valid)))
    
    if(i == 0):
        xgb_preds = xgb_pred
        lgb_preds = lgb_pred
        cat_preds = cat_pred
    else:
        xgb_preds += xgb_pred
        lgb_preds += lgb_pred
        cat_preds += cat_pred
        
xgb_pred = xgb_preds / fold    
lgb_pred = lgb_preds / fold    
cat_pred = cat_preds / fold    
    
print(xgb_pred.shape)
print(lgb_pred.shape)
print(cat_pred.shape)

test['is_churn'] = xgb_pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('xgb_sub_v2.csv', index=False)

test['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('lgb_sub_v2.csv', index=False)

test['is_churn'] = cat_pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('cat_sub_v2.csv', index=False)
    
pred = (xgb_pred + lgb_pred + cat_pred) / 3
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('ensemble_sub_v2.csv', index=False)


    
