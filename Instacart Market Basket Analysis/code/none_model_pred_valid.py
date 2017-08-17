import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

# read datasets
train_origin_df = pd.read_csv('features/none_train_datas.csv')
train_origin_df = shuffle(train_origin_df,random_state=0)

y = train_origin_df["is_none"]
order_id = train_origin_df["order_id"]
order_id = np.array(order_id)
X = np.array(train_origin_df.drop(['user_id','order_id','is_none'], axis=1),np.float32)
y = np.array(y,np.float32)
kf=KFold(n_splits=10, random_state=0, shuffle=False)      
scores=[]
order_id2pred={}
for train_index, test_index in kf.split(X):
    train, test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    order_id_train, order_id_test = order_id[train_index], order_id[test_index]
    y_mean = np.mean(y_train)
    
    import xgboost as xgb

    xgb_params = {
        'eta': 0.005,
        'max_depth': 7,
        'subsample': 0.8,
        'objective': 'reg:linear',
        'eval_metric': 'logloss',
        'base_score': y_mean,
        'silent': 1
    }

    dtrain = xgb.DMatrix(train, y_train)
    dtest = xgb.DMatrix(test, y_test)
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
    num_boost_rounds = 900
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    preds=model.predict(dtest)
    for i in range(len(order_id_test)):
        order_id2pred[order_id_test[i]]=preds[i]
    score=log_loss(y_test, preds)
    scores.append(score)
    print("Log loss score",score)
print("Avg log loss score",sum(scores)/len(scores))

out = pd.DataFrame({'order_id': list(order_id2pred.keys()), 'none_pred': list(order_id2pred.values())})
out.to_csv('none_pred.csv', index=False)