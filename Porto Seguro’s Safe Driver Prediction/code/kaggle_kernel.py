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

def all_miss(row):
    all = 0
    for v in row:
        if(v == -1):
            all += 1
    return all

def same_group_miss(row, headers, group):
    all = 0
    for i, v in enumerate(headers):
        try:
            if(v.split('_')[1] == group):
                if(row[i] == -1):
                    all += 1
        except:
            pass
            
    return all        

def same_group_sum(row, headers, group):
    sum = 0
    miss = True
    for i, v in enumerate(headers):
        try:
            if(v.split('_')[1] == group):
                if(row[i] != -1):
                    sum += row[i]
                    miss = False
        except:
            pass
            
    if(miss):
        return -1
    return sum    

groups = ['ind', 'reg', 'car', 'calc']
    
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

headers = df_test.columns.values
for group in groups:
    df_train[group + '_miss'] = df_train.apply(lambda row: same_group_miss(row, headers, group), axis = 1)
    df_train[group + '_sum'] = df_train.apply(lambda row: same_group_miss(row, headers, group), axis = 1)
    
    df_test[group + '_miss'] = df_test.apply(lambda row: same_group_miss(row, headers, group), axis = 1)
    df_test[group + '_sum'] = df_test.apply(lambda row: same_group_miss(row, headers, group), axis = 1)

df_train['all_miss'] = df_train.apply(all_miss, axis = 1)
df_test['all_miss'] = df_test.apply(all_miss, axis = 1)

df_train.to_csv("../input/pre_train.csv", index=False)   
df_test.to_csv("../input/pre_test.csv", index=False)   

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
        'max_depth': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 3228,
        'silent': True
    }


    # form DMatrices for Xgboost training
    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 2000, watchlist, early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=50)
                        
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