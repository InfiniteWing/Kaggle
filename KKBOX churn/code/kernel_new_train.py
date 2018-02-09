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
valid = pd.read_csv('train_pre_v4_201703.csv')
'''
valid = valid.drop(['is_churn'],axis=1)
df2 = pd.read_csv('../input/train_v2.csv')
print(df2.shape)
df2 = pd.merge(df2, valid, how='left', on='msno')
print(df2.shape)
valid = df2
'''
train = pd.read_csv('train_pre_2017_all_v4.csv')
'''
train['total_hours_sum'] = train['total_secs_sum'] / 3600
train['total_hours_mean'] = train['total_secs_mean'] / 3600
train['total_num_all'] = train['num_25_sum'] + train['num_50_sum'] + train['num_75_sum'] + train['num_985_sum'] + train['num_100_sum']
train['total_num_mean'] = train['total_num_all'] / train['count']
train['is_weekly_user'] = train['count'] >= 4
train['is_daily_user'] = train['count'] >= 15
'''
#train = train.drop(['last_is_churn','churn_rate','churn_count','transaction_count'],axis=1)
#train = train.drop(['transaction_date','expire_date'],axis=1)

train_v1_features = pd.read_csv('train_pre_v4_201702.csv')
train_v2_features = pd.read_csv('train_pre_v4_201703.csv')
train_v1_features = train_v1_features.drop(['is_churn','last_is_churn','churn_rate','churn_count'],axis=1)
train_v2_features = train_v2_features.drop(['is_churn','last_is_churn','churn_rate','churn_count'],axis=1)
print(train_v1_features.shape, train_v2_features.shape)
train_v1 = pd.read_csv('../input/train.csv')
train_v2 = pd.read_csv('../input/train_v2.csv')
print(train_v1.shape, train_v2.shape)
train_v1 = pd.merge(train_v1, train_v1_features, how='left', on='msno')
train_v2 = pd.merge(train_v2, train_v2_features, how='left', on='msno')
print(train_v1.shape, train_v2.shape)
train = train_v1.append(train_v2).reset_index(drop=True)

test = pd.read_csv('../input/sample_submission_v2.csv')
test_features = pd.read_csv('train_pre_v4_201704.csv')
'''
test_features['total_hours_sum'] = test_features['total_secs_sum'] / 3600
test_features['total_hours_mean'] = test_features['total_secs_mean'] / 3600
test_features['total_num_all'] = test_features['num_25_sum'] + test_features['num_50_sum'] + test_features['num_75_sum'] + test_features['num_985_sum'] + test_features['num_100_sum']
test_features['total_num_mean'] = test_features['total_num_all'] / test_features['count']
test_features['is_weekly_user'] = test_features['count'] >= 4
test_features['is_daily_user'] = test_features['count'] >= 15
'''
#test_features = test_features.drop(['is_churn','last_is_churn','churn_rate','churn_count','transaction_count'],axis=1)
#test_features = test_features.drop(['is_churn','transaction_date','expire_date'],axis=1)
test_features = test_features.drop(['is_churn'],axis=1)
test = pd.merge(test, test_features, how='left', on='msno')

members = pd.read_csv('../input/members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
valid = pd.merge(valid, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno') 
print('members merge...') 
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
valid['gender'] = valid['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(-1)
valid = valid.fillna(-1)
test = test.fillna(-1)

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
    '''
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    #model = xgb.train(params, xgb.DMatrix(x1, y1), 200,  watchlist, feval=xgb_score, maximize=False, verbose_eval=10, early_stopping_rounds=10)
    model = xgb.Booster()
    model.load_model('xgb_model_0_limit_149.model')
    #model.save_model('xgb_model_{}_limit_{}.model'.format(i, model.best_ntree_limit))
    #print(model.best_ntree_limit)
    #xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    xgb_pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=149)
    test['is_churn'] = xgb_pred.clip(0.+1e-15, 1-1e-15)
    test[['msno','is_churn']].to_csv('xgb_sub.csv', index=False)
    #xgb_valid_pred = model.predict(xgb.DMatrix(valid[cols]), ntree_limit=149)
    #valid['is_churn'] = xgb_valid_pred.clip(0.+1e-15, 1-1e-15)
    #valid[['msno','is_churn']].to_csv('xgb_valid.csv', index=False)
    '''
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
    #model = lgb.Booster(model_file='lgb_model_{}.model'.format(i))
    ax = lgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig('feature_importance_{}.png'.format(i))
    lgb_pred = model.predict(test[cols])
    model.save_model('lgb_model_{}.model'.format(i))
    #if(i == 0):
    #    pred_all = lgb_pred
    #else:
    #    pred_all += lgb_pred
        
    #lgb_pred = pred_all / fold
            
    test['is_churn'] = lgb_pred.clip(0.+1e-15, 1-1e-15)
    test[['msno','is_churn']].to_csv('lgb_sub.csv', index=False)

    '''
    lgb_valid_pred = model.predict(valid[cols])
    valid['is_churn'] = lgb_valid_pred.clip(0.+1e-15, 1-1e-15)
    
    valid[['msno','is_churn']].to_csv('lgb_valid.csv', index=False)
    for k in range(1,16):
        pred = lgb_valid_pred.clip(0.+10**-k, 1-10**-k)
        print('{}, log loss = {}'.format(k,log_loss(y2,pred)))
    total_churn = np.sum(y2)
    valid_churn = np.sum(lgb_valid_pred)
    rate = total_churn / valid_churn
    print('total churn = {}'.format(total_churn))
    print('churn rate = {}'.format(total_churn/len(y2)))
    print('predict churn rate = {}'.format(valid_churn/len(y2)))
    
    lgb_valid_pred_fix = []
    for v in lgb_valid_pred:
        if(v <0.5):
            v *= rate
        lgb_valid_pred_fix.append(v)
    lgb_valid_pred_fix = np.array(lgb_valid_pred_fix)
    print('fix log loss = {}'.format(log_loss(y2,lgb_valid_pred_fix.clip(0.+1e-15, 1-1e-15))))
    '''
    # catboost
    '''
    model = CatBoostClassifier(
        iterations=200, learning_rate=0.12,
        depth=7, l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='Logloss',
        random_seed=3228)
    #model.load_model('cat_model_{}.model'.format(i))
    model = model.fit(x1, y1,eval_set=(x2,y2))
    cat_pred = model.predict_proba(test[cols])
    model.save_model('cat_model_{}.model'.format(i))
    test['is_churn'] = cat_pred#.clip(0.+1e-15, 1-1e-15)
    test[['msno','is_churn']].to_csv('cat_sub.csv', index=False)
    cat_valid_pred = model.predict_proba(valid[cols])
    valid['is_churn'] = cat_valid_pred.clip(0.+1e-15, 1-1e-15)
    valid[['msno','is_churn']].to_csv('cat_valid.csv', index=False)
    
    pred = (xgb_pred + lgb_pred + cat_pred) / 3
    test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
    test[['msno','is_churn']].to_csv('ensemble_sub.csv', index=False)
    
    combined = np.vstack((xgb_pred,lgb_pred,cat_pred))
    pred = np.median(combined, axis = 0)
    test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
    test[['msno','is_churn']].to_csv('ensemble_median_sub.csv', index=False)
    '''
    
    '''
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    '''
'''
pred /= fold
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('5fold_sub.csv', index=False)
#test[['msno','is_churn']].to_csv('submission.csv.gz', index=False, compression='gzip')
'''