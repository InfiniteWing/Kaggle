#
# Source: https://www.kaggle.com/seesee/concise-catboost-starter-ensemble-plb-0-06435/output
# Revised by InfiniteWing
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime as dt

properties = pd.read_csv('../input/properties_2017.csv', low_memory = False)
sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)

train2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('../input/train_train_2017.csv', parse_dates=['transactiondate'], low_memory=False)
train_df = pd.concat([train2016, train2017], ignore_index=True)
valid2017 = pd.read_csv('../input/train_valid_2017.csv', parse_dates=['transactiondate'], low_memory=False)
def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


train_df = add_date_features(train_df)
train_df = train_df.merge(properties, how='left', on='parcelid')
valid2017 = valid2017.merge(properties, how='left', on='parcelid')
test_df = pd.merge(sample_submission[['ParcelId']], properties.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')

del properties, train2016, train2017
gc.collect();

print("Train: ", train_df.shape)
print("Test: ", test_df.shape)
print("Removing outliers")

train_df = train_df[ train_df.logerror > -0.4 ]
train_df = train_df[ train_df.logerror < 0.4 ]

print("Train: ", train_df.shape)

print('Remove missing data fields ...')

missing_perc_thresh = 0.98
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)
print("We exclude: %s" % len(exclude_missing))

del num_rows, missing_perc_thresh
gc.collect();

print ("Remove features with one unique value !!")
exclude_unique = []
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0:
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)
print("We exclude: %s" % len(exclude_unique))

print ("Define training features !!")
exclude_other = ['parcelid', 'logerror','propertyzoningdesc']
train_features = []
for c in train_df.columns:
    if c not in exclude_missing \
       and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)
print("We use these for training: %s" % len(train_features))

print ("Define categorial features !!")
cat_feature_inds = []
cat_unique_thresh = 1000
for i, c in enumerate(train_features):
    num_uniques = len(train_df[c].unique())
    if num_uniques < cat_unique_thresh \
       and not 'sqft' in c \
       and not 'cnt' in c \
       and not 'nbr' in c \
       and not 'number' in c:
        cat_feature_inds.append(i)
        
print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])

print ("Replacing NaN values by -999 !!")
train_df.fillna(-999, inplace=True)
valid2017.fillna(-999, inplace=True)

print ("Training time !!")
X_train = train_df[train_features]
y_train = train_df.logerror
print(X_train.shape, y_train.shape)

valid_times = ['2017-08-01']

from sklearn.metrics import mean_absolute_error
num_ensembles = 5
valid_scores = 0.0
y_preds = None
for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations=630, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    model.fit(
        X_train, y_train,
        cat_features=cat_feature_inds)
    for j, v in enumerate(valid_times):
        valid2017['transactiondate'] = pd.Timestamp(v)  # Dummy
        valid2017 = add_date_features(valid2017)
        X_test = valid2017[train_features]
        y_valid = valid2017.logerror
        pred = model.predict(X_test)
        valid_score = mean_absolute_error(y_valid, pred)
        print('Valid MAE =', valid_score)
        valid_scores += valid_score
        if(i == 0):
            y_preds = pred
        else:
            y_preds += pred
            
valid_scores /= num_ensembles
y_preds /= num_ensembles
print('Average valid MAE =', valid_scores)


valid_score = mean_absolute_error(y_valid, y_preds)

print('Blending valid MAE =', valid_score)

output = pd.DataFrame({'valid_pred': y_preds})
output.to_csv('valid_catboost.csv', index=False, float_format = '%.5f')
                  