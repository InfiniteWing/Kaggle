import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from sklearn.decomposition import PCA, FastICA
# read datasets
X = pd.read_csv('train.csv/train.csv')
X = shuffle(X,random_state=0)


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


y = X["y"]
X = np.array(X.drop(['y','ID'], axis=1),np.float32)
y = np.array(y,np.float32)
kf=KFold(n_splits=10, random_state=0, shuffle=True)
scores=[]
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    train, test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_mean = np.mean(y_train)
    drop_indexs=[]
    y_train_exclude_outlier=[]
    train_exclude_outlier=[]
    for i,v in enumerate(y_train):
        if(v>145 or v<74):
            drop_indexs.append(i)
        else:
            train_exclude_outlier.append(train[i])
            y_train_exclude_outlier.append(v)
    train=np.array(train_exclude_outlier)
    y_train=np.array(y_train_exclude_outlier)
    # shape        
    print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

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
    dtest = xgb.DMatrix(test, y_test)
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
    #print(num_boost_rounds)

    # train model
    import matplotlib
    model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
    #model=SVR(kernel='linear',C=0.001)
    #model=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
    #  normalize=False, random_state=0, solver='auto', tol=0.001)
    #model.fit(train, y_train) 
    # check f2-score (to get higher score - increase num_boost_round in previous cell)
    from sklearn.metrics import r2_score

    # now fixed, correct calculation
    score=r2_score(dtest.get_label(), model.predict(dtest))
    print(score)
    scores.append(score)
print("AVG R^2 score = {}".format(sum(scores)/len(scores)))