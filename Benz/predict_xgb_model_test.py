import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from sklearn import ensemble
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
    
    from sklearn import linear_model
    from sklearn import neural_network,neighbors
    from sklearn import svm
    from sklearn import kernel_ridge
    model = ensemble.RandomForestRegressor(n_estimators=18,random_state=0)
    #model=neural_network.MLPRegressor(random_state=0,solver='lbfgs')
    #model=neighbors.KNeighborsRegressor(n_neighbors=50)
    #model=kernel_ridge.KernelRidge()
    #model=svm.SVR(C=5,kernel='linear')
    model.fit(train,y_train)
    
    from sklearn.metrics import r2_score

    # now fixed, correct calculation
    score=r2_score(y_test, model.predict(test))
    print(score)
    scores.append(score)
print("AVG R^2 score = {}".format(sum(scores)/len(scores)))