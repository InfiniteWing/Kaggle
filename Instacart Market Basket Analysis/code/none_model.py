import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# read datasets
train = pd.read_csv('features/none_train_datas.csv')
test=pd.read_csv('features/none_test_datas.csv')
order_id=test['order_id']

y_train = train["is_none"]
y_mean = np.mean(y_train)
       
print('Shape train: {}\n Shape test: {}'.format(train.shape,test.shape))

import xgboost as xgb

xgb_params = {
    'eta': 0.005,
    'max_depth': 6,
    'subsample': 0.8,
    'objective': 'reg:linear',
    'eval_metric': 'logloss',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(train.drop(['user_id','order_id','is_none'], axis=1), y_train)
dtest = xgb.DMatrix(test.drop(['user_id','order_id'], axis=1))
'''
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   nfold=10,
                   num_boost_round=1500, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )
'''
#num_boost_rounds = len(cv_result)
#print(num_boost_rounds)
num_boost_rounds=800
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
preds=model.predict(dtest)

out = pd.DataFrame({'order_id': order_id, 'none_pred': preds})
out.to_csv('test_none_pred.csv', index=False)