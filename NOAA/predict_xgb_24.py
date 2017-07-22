import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

bit=24

train = pd.read_csv('pre/train_pre_origin_{}.csv'.format(bit))
train1 = pd.read_csv('pre/train_subadult_males_pre_origin_{}.csv'.format(bit))
train2 = pd.read_csv('pre/train_adult_females_pre_origin_{}.csv'.format(bit))
train3 = pd.read_csv('pre/train_juveniles_pre_origin_{}.csv'.format(bit))
train4 = pd.read_csv('pre/train_pups_pre_origin_{}.csv'.format(bit))
y_train = train["adult_males"].values.astype(np.int32)
y_train1 = train1["adult_males"].values.astype(np.int32)
y_train2 = train2["adult_males"].values.astype(np.int32)
y_train3 = train3["adult_males"].values.astype(np.int32)
y_train4 = train4["adult_males"].values.astype(np.int32)

n_comp = 16

from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


# PCA
pca = PCA(n_components=n_comp, random_state=0)
pca2_results_train = pca.fit_transform(train.drop(["adult_males"], axis=1))

# ICA
ica = FastICA(n_components=n_comp, random_state=0)
ica2_results_train = ica.fit_transform(train.drop(["adult_males"], axis=1))


# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    train1['pca_' + str(i)] = pca2_results_train[:,i-1]
    train1['ica_' + str(i)] = ica2_results_train[:,i-1]
    train2['pca_' + str(i)] = pca2_results_train[:,i-1]
    train2['ica_' + str(i)] = ica2_results_train[:,i-1]
    train3['pca_' + str(i)] = pca2_results_train[:,i-1]
    train3['ica_' + str(i)] = ica2_results_train[:,i-1]
    train4['pca_' + str(i)] = pca2_results_train[:,i-1]
    train4['ica_' + str(i)] = ica2_results_train[:,i-1]
y_mean = np.mean(y_train)

# mmm, xgboost, loved by everyone ^-^
import xgboost as xgb

# prepare dict of params for xgboost to run with
xgb_params = {
    'eta': 0.01,
    'max_depth': 5,
    'subsample': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train.drop(["adult_males"], axis=1), y_train)
dtrain1 = xgb.DMatrix(train1.drop(["adult_males"], axis=1), y_train1)
dtrain2 = xgb.DMatrix(train2.drop(["adult_males"], axis=1), y_train2)
dtrain3 = xgb.DMatrix(train3.drop(["adult_males"], axis=1), y_train3)
dtrain4 = xgb.DMatrix(train4.drop(["adult_males"], axis=1), y_train4)
'''
# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain2, 
                   nfold=5,
                   num_boost_round=200, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)
'''
num_boost_rounds=60

# train model
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds)
model1 = xgb.train(dict(xgb_params, silent=1), dtrain1, num_boost_round=num_boost_rounds)
model2 = xgb.train(dict(xgb_params, silent=1), dtrain2, num_boost_round=num_boost_rounds)
model3 = xgb.train(dict(xgb_params, silent=1), dtrain3, num_boost_round=num_boost_rounds)
model4 = xgb.train(dict(xgb_params, silent=1), dtrain4, num_boost_round=num_boost_rounds)
for test_index in range(6):
    # read datasets
    test = pd.read_csv("pre/test_pre_origin_{}_{}.csv".format(bit,test_index))
    
    pca2_results_test = pca.transform(test.drop(["test_id"],axis=1))
    ica2_results_test = ica.transform(test.drop(["test_id"],axis=1))

    # Append decomposition components to datasets
    for i in range(1, n_comp+1):
        test['pca_' + str(i)] = pca2_results_test[:, i-1]
        test['ica_' + str(i)] = ica2_results_test[:, i-1]

    dtest = xgb.DMatrix(test.drop(["test_id"], axis=1))

    pred = model.predict(dtest)
    pred1 = model1.predict(dtest)
    pred2 = model2.predict(dtest)
    pred3 = model3.predict(dtest)
    pred4 = model4.predict(dtest)
    y_pred=[]
    y_pred1=[]
    y_pred2=[]
    y_pred3=[]
    y_pred4=[]
    for i,predict in enumerate(pred):
        y_pred.append(int(pred[i]*0.9+0.5))
        y_pred1.append(int(pred1[i]*0.9+0.5))
        y_pred2.append(int(pred2[i]*0.9+0.5))
        y_pred3.append(int(pred3[i]*0.9+0.5))
        y_pred4.append(int(pred4[i]*0.9+0.5))
    y_pred=np.array(y_pred)
    y_pred1=np.array(y_pred1)
    y_pred2=np.array(y_pred2)
    y_pred3=np.array(y_pred3)
    y_pred4=np.array(y_pred4)
    output = pd.DataFrame({'test_id': test['test_id'].astype(np.int32),
            'adult_males': y_pred.astype(np.int32),
            'subadult_males': y_pred1.astype(np.int32),
            'adult_females': y_pred2.astype(np.int32),
            'juveniles': y_pred3.astype(np.int32),
            'pups': y_pred4.astype(np.int32)})
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    from datetime import datetime
    output.to_csv('sub_{}_{}.csv'.format(bit,test_index), index=False)
