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
    
def gini_score(preds, labels):
    return gini_normalized(labels, preds)
    
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


target_train = df_train['target'].values
id_test = df_test['id'].values

train = np.array(df_train.drop(['target','id','ps_reg_03', \
                'ps_car_03_cat','ps_car_05_cat','ps_car_14'], axis = 1))
test = np.array(df_test.drop(['id','ps_reg_03', \
                'ps_car_03_cat','ps_car_05_cat','ps_car_14'], axis = 1))

xgb_preds = []
valid_scores = 0.0

K = 5
kf = KFold(n_splits = K, random_state = 3228, shuffle = True)

fold = 1

for train_index, test_index in kf.split(train):
    train_X, valid_X = train[train_index], train[test_index]
    train_y, valid_y = target_train[train_index], target_train[test_index]
        
    # prepare dict of params for xgboost to run with
    # params configuration also from anokas' kernel
    xgb_params = {
        'eta': 0.02,
        'max_depth': 6,
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
    #model = xgb.train(xgb_params, d_train, 2000, watchlist, early_stopping_rounds=100, feval=gini_xgb, maximize=True, verbose_eval=50)
    model = xgb.train(xgb_params, d_train, num_boost_round = 666)     
    valid_pred = model.predict(d_valid)
    valid_score = gini_score(valid_pred, d_valid.get_label())
    print("Fold {}, valid gini score = {}".format(fold, valid_score))
    fold += 1
    valid_scores += float(valid_score)
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
    
preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(K):
        sum+=xgb_preds[j][i]
    preds.append(sum / K)

print("CV average score = {}".format(valid_scores / K))

output = pd.DataFrame({'id': id_test, 'target': preds})
output.to_csv("10foldCV_sub.csv", index=False)   