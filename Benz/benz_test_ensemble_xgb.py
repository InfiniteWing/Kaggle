import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FastICA
from sklearn import linear_model
from sklearn import neural_network,neighbors
from sklearn import svm
from sklearn import kernel_ridge
from sklearn import ensemble

# read datasets
X = pd.read_csv('train.csv/train.csv')
X = shuffle(X,random_state=0)
'''
# process columns, apply LabelEncoder to categorical features

for c in X.columns:
    if X[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(X[c].values)) 
        X[c] = lbl.transform(list(X[c].values))
'''
# source: https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554
cat_cols = []
for c in X.columns:
    if X[c].dtype == 'object':
        cat_cols.append(c)
#print('Categorical columns:', cat_cols)
# Duplicate features
d = {}; done = []
cols = X.columns.values
for c in cols: d[c]=[]
for i in range(len(cols)):
    if i not in done:
        for j in range(i+1, len(cols)):
            if all(X[cols[i]] == X[cols[j]]):
                done.append(j)
                d[cols[i]].append(cols[j])
dub_cols = []
for k in d.keys():
    if len(d[k]) > 0: 
        # print k, d[k]
        dub_cols += d[k]        
#print('Dublicates:', dub_cols)
# Constant columns
const_cols = []
for c in cols:
    if len(X[c].unique()) == 1:
        const_cols.append(c)
#print('Constant cols:', const_cols)
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

# end source from https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554
 
pred_feature_id=[]
y = X["y"]
ids=np.array(X['ID'])
X = np.array(X.drop(['y','ID'], axis=1),np.float32)
y = np.array(y,np.float32)
kf=KFold(n_splits=10, random_state=0, shuffle=True)
ensemble_scores=[]
xgb_scores=[]
lgb_scores=[]

ensemble_preds=[]
xgb_preds=[]
lgb_preds=[]
bridge_preds=[]
ridge_preds=[]
labels=[]

for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train, test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    id_train,id_test = ids[train_index], ids[test_index]
    y_mean = np.mean(y_train)
    drop_indexs=[]
    y_train_exclude_outlier=[]
    train_exclude_outlier=[]
    for i,v in enumerate(y_train):
        if(v>155):
            drop_indexs.append(i)
        else:
            train_exclude_outlier.append(train[i])
            y_train_exclude_outlier.append(v)
    train=np.array(train_exclude_outlier)
    y_train=np.array(y_train_exclude_outlier)
    # shape        
    print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
    ### Regressor
    import json
    import lightgbm as lgb
    # create dataset for lightgbm
    
    lgb_train = lgb.Dataset(train, y_train)
    lgb_test = lgb.Dataset(test)
    
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
    dtrain = xgb.DMatrix(train, y_train)
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
    xgb_pred=model.predict(dtest)
    
    
    
    model = linear_model.Ridge(random_state=0,alpha=50)
    model.fit(train,y_train)
    ridge_pred=model.predict(test)
    
    model = linear_model.BayesianRidge()
    model.fit(train,y_train)
    bridge_pred=model.predict(test)
    
    #model=ensemble.RandomForestRegressor()
    #model.fit(train,y_train)
    #forest_pred=model.predict(test)
    
    from sklearn.metrics import r2_score
    lgb_score=r2_score(y_test, lgb_pred)
    print("LightGBM R^2 score = {}".format(lgb_score))
    xgb_score=r2_score(y_test, xgb_pred)
    print("XGBoost R^2 score = {}".format(xgb_score))
    ridge_score=r2_score(y_test, ridge_pred)
    print("Ridge R^2 score = {}".format(ridge_score))
    bridge_score=r2_score(y_test, bridge_pred)
    print("BayesianRidge R^2 score = {}".format(bridge_score))
    #forest_score=r2_score(y_test, forest_pred)
    #print("RandomForest R^2 score = {}".format(forest_score))
    ensemble_pred=(lgb_pred*0.55+xgb_pred*0.25+ridge_pred*0.10+bridge_pred*0.10)
    ensemble_score=r2_score(y_test, ensemble_pred)
    print("ensemble R^2 score = {}".format(ensemble_score))
    ensemble_preds+=list(ensemble_pred)
    xgb_preds+=list(xgb_pred)
    lgb_preds+=list(lgb_pred)
    ridge_preds+=list(ridge_pred)
    bridge_preds+=list(bridge_pred)
    pred_feature_id+=list(id_test)
    labels+=list(y_test)
    ensemble_scores.append(ensemble_score)
    xgb_scores.append(xgb_score)
    lgb_scores.append(lgb_score)
print(len(xgb_preds))
print(len(labels))
print("AVG LightGBM R^2 score = {}".format(r2_score(labels, lgb_preds)))
print("AVG XGBoost R^2 score = {}".format(r2_score(labels, xgb_preds)))
print("AVG ensemble R^2 score = {}".format(r2_score(labels, ensemble_preds)))
output = pd.DataFrame({'id': pred_feature_id,'y': labels, 'ensemble_y': ensemble_preds
            , 'xgb_y': xgb_preds, 'lgb_y': lgb_preds, 'ridge_y': ridge_preds, 'bridge_y': bridge_preds
            })
output.to_csv("ensemble_feature.csv",index=False)   