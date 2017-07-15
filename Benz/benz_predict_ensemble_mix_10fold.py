import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import linear_model
from sklearn import neural_network,neighbors
from sklearn import svm
from sklearn import kernel_ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# read datasets
train = pd.read_csv('train.csv/train.csv')
test = pd.read_csv('test.csv/test.csv')

# source: https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554

cat_cols = []
for c in train.columns:
    if train[c].dtype == 'object':
        cat_cols.append(c)
print('Categorical columns:', cat_cols)
# Duplicate features
d = {}; done = []
cols = train.columns.values
for c in cols: d[c]=[]
for i in range(len(cols)):
    if i not in done:
        for j in range(i+1, len(cols)):
            if all(train[cols[i]] == train[cols[j]]):
                done.append(j)
                d[cols[i]].append(cols[j])
dub_cols = []
for k in d.keys():
    if len(d[k]) > 0: 
        # print k, d[k]
        dub_cols += d[k]        
print('Dublicates:', dub_cols)
# Constant columns
const_cols = []
for c in cols:
    if len(train[c].unique()) == 1:
        const_cols.append(c)
print('Constant cols:', const_cols)

train['eval_set'] = 0;
test['eval_set'] = 1;
X = pd.concat([train, test], axis=0, copy=True)
# Reset index
X=X.reset_index(drop=True)

def add_new_col(x):
    if x not in new_col.keys(): 
        # set n/2 x if is contained in test, but not in train 
        # (n is the number of unique labels in train)
        # or an alternative could be -100 (something out of range [0; n-1]
        return int(len(new_col.keys())/2)
    return new_col[x] # rank of the label

for c in cat_cols:
    # get labels and corresponding means
    new_col = X.groupby(c).y.mean().sort_values().reset_index()
    # make a dictionary, where key is a label and value is the rank of that label
    new_col = new_col.reset_index().set_index(c).drop('y', axis=1)['index'].to_dict()
    # add new column to the dataframe
    X[c + '_new'] = X[c].apply(add_new_col)
X=X.drop(list((set(const_cols) | set(dub_cols) | set(cat_cols))), axis=1)

# Train
train = X[X.eval_set == 0]
y=train['y']
train = train.drop(['eval_set','ID','y'], axis=1)
train=np.array(train)

# Test
test_All = X[X.eval_set == 1]
test = test_All.drop(['y','ID', 'eval_set'], axis=1)
test=np.array(test)
# end source from https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554 


kf=KFold(n_splits=10, random_state=0, shuffle=True)

ensemble_preds=[]
xgb_preds=[]
lgb_preds=[]
bridge_preds=[]
ridge_preds=[]

ensemble_valid_preds=[]
xgb_valid_preds=[]
lgb_valid_preds=[]
bridge_valid_preds=[]
ridge_valid_preds=[]

labels=[]

for train_index, test_index in kf.split(train):
    train_X, valid_X = train[train_index], train[test_index]
    train_y, valid_y = y[train_index], y[test_index]
    y_mean = np.mean(train_y)
    drop_indexs=[]
    y_train_exclude_outlier=[]
    train_exclude_outlier=[]
    for i,v in enumerate(train_y):
        if(v>155):
            drop_indexs.append(i)
        else:
            train_exclude_outlier.append(train_X[i])
            y_train_exclude_outlier.append(v)
    train_X=np.array(train_exclude_outlier)
    train_y=np.array(y_train_exclude_outlier)
    # shape        
    print('Shape train: {}\nShape valid: {}\nShape test: {}'.format(train_X.shape,valid_X.shape, test.shape))
    ### Regressor
    import json
    import lightgbm as lgb
    # create dataset for lightgbm
    
    lgb_train = lgb.Dataset(train_X, train_y)
    
    # train
    n_round=500

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2_root', 'rmse'},
        'num_leaves': 6,
        'max_depth': 4,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.95,
        'bagging_freq': 5,
        'verbose': 0,
        'max_bin':4
    }
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=n_round)
                    
    # predict
    lgb_valid_pred = gbm.predict(valid_X)
    lgb_pred = gbm.predict(test)
    
    # mmm, xgboost, loved by everyone ^-^
    import xgboost as xgb

    # prepare dict of params for xgboost to run with
    xgb_params = {
        'eta': 0.005,
        'max_depth': 4,
        'subsample': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'base_score': y_mean, # base prediction = mean(target)
        'silent': 1
    }

    # form DMatrices for Xgboost training
    dtrain = xgb.DMatrix(train_X, train_y)
    dvalid = xgb.DMatrix(valid_X)
    dtest = xgb.DMatrix(test)
    # xgboost, cross-validation
    '''
    cv_result = xgb.cv(xgb_params, 
                       dtrain, 
                       num_boost_round=500, # increase to have better results (~700)
                       early_stopping_rounds=50,
                       verbose_eval=50, 
                       show_stdv=False
                      )

    num_boost_rounds = len(cv_result)
    '''
    num_boost_rounds=800
    import matplotlib
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    xgb_valid_pred=model.predict(dvalid)
    xgb_pred=model.predict(dtest)
    
    
    
    model = linear_model.Ridge(random_state=0,alpha=50)
    model.fit(train_X,train_y)
    ridge_valid_pred=model.predict(valid_X)
    ridge_pred=model.predict(test)
    
    model = linear_model.BayesianRidge()
    model.fit(train_X,train_y)
    bridge_valid_pred=model.predict(valid_X)
    bridge_pred=model.predict(test)
    
    
    from sklearn.metrics import r2_score
    lgb_score=r2_score(valid_y, lgb_valid_pred)
    print("LightGBM R^2 score = {}".format(lgb_score))
    xgb_score=r2_score(valid_y, xgb_valid_pred)
    print("XGBoost R^2 score = {}".format(xgb_score))
    ridge_score=r2_score(valid_y, ridge_valid_pred)
    print("Ridge R^2 score = {}".format(ridge_score))
    bridge_score=r2_score(valid_y, bridge_valid_pred)
    print("BayesianRidge R^2 score = {}".format(bridge_score))
    
    #ensemble_valid_pred=(lgb_valid_pred*0.6+xgb_valid_pred*0.2+ridge_valid_pred*0.1+bridge_valid_pred*0.1)
    ensemble_valid_pred=(lgb_valid_pred*0.7+xgb_valid_pred*0.2+ridge_valid_pred*0.05+bridge_valid_pred*0.05)
    ensemble_score=r2_score(valid_y, ensemble_valid_pred)
    
    ensemble_pred=(lgb_pred*0.6+xgb_pred*0.2+ridge_pred*0.1+bridge_pred*0.1)
    #ensemble_pred=(lgb_pred*0.70+xgb_pred*0.20+ridge_pred*0.05+bridge_pred*0.05)
    print("ensemble R^2 score = {}".format(ensemble_score))
    
    ensemble_valid_preds+=list(ensemble_valid_pred)
    xgb_valid_preds+=list(xgb_valid_pred)
    lgb_valid_preds+=list(lgb_valid_pred)
    ridge_valid_preds+=list(ridge_valid_pred)
    bridge_valid_preds+=list(bridge_valid_pred)
    labels+=list(valid_y)
    
    ensemble_preds.append(list(ensemble_pred))
    xgb_preds.append(list(xgb_pred))
    lgb_preds.append(list(lgb_pred))
    ridge_preds.append(list(ridge_pred))
    bridge_preds.append(list(bridge_pred))
    
    
print(len(xgb_preds))
print(len(labels))
print("AVG LightGBM R^2 score = {}".format(r2_score(labels, lgb_valid_preds)))
print("AVG XGBoost R^2 score = {}".format(r2_score(labels, xgb_valid_preds)))
print("AVG ensemble R^2 score = {}".format(r2_score(labels, ensemble_valid_preds)))

preds=[]
print(len(ensemble_preds[0]))
for i in range(len(ensemble_preds[0])):
    sum=0
    for j in range(10):
        try:
            sum+=ensemble_preds[j][i]
        except:
            print(i,j)
    preds.append(sum/10)

output = pd.DataFrame({'id': test_All['ID'],'y': preds})
output.to_csv("submission_10fold_ensembling.csv",index=False)   