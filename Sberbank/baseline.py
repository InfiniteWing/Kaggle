#
# This script is inspired by this kernal:
# https://www.kaggle.com/paulorzp/xgb-mix-models-lb-0-31
#

import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import warnings
warnings.filterwarnings("ignore")

#load files
train = pd.read_csv('train.csv', parse_dates=['timestamp'])
test = pd.read_csv('test.csv', parse_dates=['timestamp'])
id_test = test['id']

#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = train.loc[bad_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = test.loc[bad_index, "full_sq"]
bad_index = train[train.life_sq < 3.31].index
train.loc[bad_index, "life_sq"] = 3.31
bad_index = test[test.life_sq < 3.31].index
test.loc[bad_index, "life_sq"] = 3.31
bad_index = train[train.full_sq < 3.31].index
train.loc[bad_index, "full_sq"] = 3.31
bad_index = test[test.full_sq < 3.31].index
test.loc[bad_index, "full_sq"] = 3.31
kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1900].index
train.loc[bad_index, "build_year"] = 1900
bad_index = test[test.build_year < 1900].index
test.loc[bad_index, "build_year"] = 1900
bad_index = train[train.build_year >2015].index
train.loc[bad_index, "build_year"] = 2015
bad_index = test[test.build_year > 2015].index
test.loc[bad_index, "build_year"] = 2015
bad_index = train[train.num_room == 0].index
for index in bad_index:
    if(not np.isnan(train['life_sq'][index])):
        train['num_room'][index]=int(train['life_sq'][index]/20)
bad_index = test[test.num_room == 0].index
for index in bad_index:
    if(not np.isnan(test['life_sq'][index])):
        test['num_room'][index]=int(test['life_sq'][index]/20)

bad_index = train[train.num_room > train.life_sq/3.31].index
for index in bad_index:
    if(not np.isnan(train['life_sq'][index])):
        train['num_room'][index]=int(train['life_sq'][index]/3.31)
bad_index = test[test.num_room > test.life_sq/3.31].index
for index in bad_index:
    if(not np.isnan(test['life_sq'][index])):
        test['num_room'][index]=int(test['life_sq'][index]/3.31)    



bad_index = train[train.floor > train.max_floor].index
for index in bad_index:
    train['floor'][index]=train['max_floor'][index]
bad_index = test[test.floor > test.max_floor].index
for index in bad_index:
    test['floor'][index]=test['max_floor'][index]
    
bad_index = train[train.floor == 0].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = test[test.floor == 0].index
test.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, ["max_floor", "floor"]] = np.NaN

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 500000]
train = train[train.price_doc/train.full_sq >= 8000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

y_train = train["price_doc"]
id_train = train['id']

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)
num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))

x_train = x_all[:num_train]
x_test = x_all[num_train:]


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_result=xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,verbose_eval=20, show_stdv=False)
num_boost_rounds = len(cv_result) 

print('XGBoost training...')
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_rounds, verbose_eval=False)
y_predict = model.predict(dtest)
result = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

result.to_csv('submission.csv', index=False)