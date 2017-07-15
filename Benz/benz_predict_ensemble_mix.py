import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn import linear_model
from sklearn import neural_network,neighbors
from sklearn import svm
from sklearn import kernel_ridge

# read datasets
train = pd.read_csv('train.csv/train.csv')
test = pd.read_csv('test.csv/test.csv')
'''
# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
'''
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
train = train.drop(['eval_set'], axis=1)

# Test
test = X[X.eval_set == 1]
test = test.drop(['y', 'eval_set'], axis=1)

# end source from https://www.kaggle.com/yohanb/categorical-features-encoding-xgb-0-554 
        
        
y_train=train['y']
drop_indexs=[]
y_train_exclude_outlier=[]
train_exclude_outlier=[]
for i,v in enumerate(y_train):
    if(v>155):
        drop_indexs.append(i)
        
train=train.drop(drop_indexs)

y_train = train["y"]
y_mean = np.mean(y_train)
id_test = test['ID'].astype(np.int32)
# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

train1 = np.array(train.drop(['y','ID'], axis=1),np.float32)
y_train1 = np.array(y_train,np.float32)
test1 = np.array(test.drop(['ID'], axis=1),np.float32)

lgb_train = lgb.Dataset(train1, y_train1)
lgb_test = lgb.Dataset(test1)

n_round=500

params = {
        'task': 'train',
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

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=n_round)
from sklearn.metrics import r2_score
# now fixed, correct calculation
print(r2_score(y_train, gbm.predict(train1)))

print('Start predicting...')
# predict
lgb_pred = gbm.predict(test1)


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
dtrain = xgb.DMatrix(train.drop(['y','ID'], axis=1), y_train)
dtest = xgb.DMatrix(test.drop(['ID'], axis=1))
# xgboost, cross-validation
'''
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=800, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )
'''
num_boost_rounds = 800#len(cv_result)
print(num_boost_rounds)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)

# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# make predictions and save results
xgb_pred = model.predict(dtest)

X=np.array(train.drop(['y','ID'], axis=1))
y=y_train
X_test=np.array(test.drop(['ID'], axis=1))
model = linear_model.Ridge(random_state=0,alpha=50)
model.fit(X,y)
ridge_pred=model.predict(X_test)
    
model = linear_model.BayesianRidge()
model.fit(X,y)
bridge_pred=model.predict(X_test)


ensemble_pred=(lgb_pred*0.6+xgb_pred*0.2+ridge_pred*0.1+bridge_pred*0.1)

output = pd.DataFrame({'id': id_test, 'y': ensemble_pred})
output.to_csv('Benz_submission_ensemble_v2.csv', index=False)
