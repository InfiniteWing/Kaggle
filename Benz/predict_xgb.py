import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
# read datasets
train = pd.read_csv('train.csv/train.csv')
test = pd.read_csv('test.csv/test.csv')

# process columns, apply LabelEncoder to categorical features
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder() 
        lbl.fit(list(train[c].values) + list(test[c].values)) 
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

drop_indexs=[]
y_train = train["y"]
for index,v in enumerate(y_train):
    if(v>145 or v<74):
        drop_indexs.append(index)
train=train.drop(drop_indexs)

y_train = train["y"]
y_mean = np.mean(y_train)
# shape        
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

# Append decomposition components to datasets
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

# train model
import matplotlib
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
'''
xgb.plot_importance(model)
importance = model.get_fscore()
print(len(importance))
importance=sorted(importance, key=importance.get)
for key,value in enumerate(importance):
    print(key,value)
matplotlib.pyplot.show()
'''
# check f2-score (to get higher score - increase num_boost_round in previous cell)
from sklearn.metrics import r2_score

# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# make predictions and save results
from datetime import datetime
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('submission_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)