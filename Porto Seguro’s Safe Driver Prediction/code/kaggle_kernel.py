import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb
import statistics

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]
    
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

target_train = df_train['target'].values
id_test = df_test['id'].values

train = np.array(df_train.drop(['target','id'], axis = 1))
test = np.array(df_test.drop(['id'], axis = 1))

xgb_preds = []

K = 10
kf = KFold(n_splits = K, random_state = 3228, shuffle = True)

for train_index, test_index in kf.split(train):
    train_X, valid_X = train[train_index], train[test_index]
    train_y, valid_y = target_train[train_index], target_train[test_index]
    
    # shape        
    print('Shape train: {}\nShape valid: {}\nShape test: {}'.format(train_X.shape, valid_X.shape, test.shape))

    # prepare dict of params for xgboost to run with
    # params configuration also from anokas' kernel
    xgb_params = {
        'eta': 0.02,
        'max_depth': 6,
        'subsample': 0.9,
        'objective': 'binary:logistic',
        'silent': 1,
        'colsample_bytree': 0.9
    }

    # form DMatrices for Xgboost training
    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(test)
    
    model = xgb.train(xgb_params, d_train, num_boost_round = 600)
                        
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
    
preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(K):
        sum+=xgb_preds[j][i]
    preds.append(sum / K)

output = pd.DataFrame({'id': id_test, 'target': preds})
output.to_csv("10foldCV_sub.csv", index=False)   