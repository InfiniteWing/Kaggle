# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

import gc

import datetime as dt

from sklearn.model_selection import KFold





# Source https://www.kaggle.com/nikunjm88/creating-additional-features

def feature_engineering(df_train):

    #life of property



    df_train['N-life'] = 2018 - df_train['yearbuilt']

    #error in calculation of the finished living area of home

    df_train['N-LivingAreaError'] = df_train['calculatedfinishedsquarefeet']/df_train['finishedsquarefeet12']



    #proportion of living area



    df_train['N-LivingAreaProp'] = df_train['calculatedfinishedsquarefeet']/df_train['lotsizesquarefeet']



    df_train['N-LivingAreaProp2'] = df_train['finishedsquarefeet12']/df_train['finishedsquarefeet15']



    #Amout of extra space



    df_train['N-ExtraSpace'] = df_train['lotsizesquarefeet'] - df_train['calculatedfinishedsquarefeet'] 



    df_train['N-ExtraSpace-2'] = df_train['finishedsquarefeet15'] - df_train['finishedsquarefeet12'] 



    #Total number of rooms



    df_train['N-TotalRooms'] = df_train['bathroomcnt']*df_train['bedroomcnt']



    #Average room size



    df_train['N-AvRoomSize'] = df_train['calculatedfinishedsquarefeet']/df_train['roomcnt'] 



    # Number of Extra rooms



    df_train['N-ExtraRooms'] = df_train['roomcnt'] - df_train['N-TotalRooms'] 



    #Ratio of the built structure value to land area



    df_train['N-ValueProp'] = df_train['structuretaxvaluedollarcnt']/df_train['landtaxvaluedollarcnt']



    #Does property have a garage, pool or hot tub and AC?



    df_train['N-GarPoolAC'] = ((df_train['garagecarcnt']>0) & (df_train['pooltypeid10']>0) & (df_train['airconditioningtypeid']!=5))*1 



    df_train["N-location"] = df_train["latitude"] + df_train["longitude"]

    df_train["N-location-2"] = df_train["latitude"]*df_train["longitude"]

    df_train["N-location-2round"] = df_train["N-location-2"].round(-4)

    df_train["N-latitude-round"] = df_train["latitude"].round(-4)

    df_train["N-longitude-round"] = df_train["longitude"].round(-4)

     

    #Ratio of tax of property over parcel



    df_train['N-ValueRatio'] = df_train['taxvaluedollarcnt']/df_train['taxamount']

    #TotalTaxScore



    df_train['N-TaxScore'] = df_train['taxvaluedollarcnt']*df_train['taxamount']



    #polnomials of tax delinquency year



    df_train["N-taxdelinquencyyear-2"] = df_train["taxdelinquencyyear"] ** 2



    df_train["N-taxdelinquencyyear-3"] = df_train["taxdelinquencyyear"] ** 3



    #Length of time since unpaid taxes



    df_train['N-life'] = 2018 - df_train['taxdelinquencyyear']

    #Number of properties in the zip



    zip_count = df_train['regionidzip'].value_counts().to_dict()



    df_train['N-zip_count'] = df_train['regionidzip'].map(zip_count)

    #Number of properties in the city



    city_count = df_train['regionidcity'].value_counts().to_dict()



    df_train['N-city_count'] = df_train['regionidcity'].map(city_count)

    #Number of properties in the city



    region_count = df_train['regionidcounty'].value_counts().to_dict()



    df_train['N-county_count'] = df_train['regionidcounty'].map(city_count)

    #Indicator whether it has AC or not



    df_train['N-ACInd'] = (df_train['airconditioningtypeid']!=5)*1

    #Indicator whether it has Heating or not 



    df_train['N-HeatInd'] = (df_train['heatingorsystemtypeid']!=13)*1







    #There's 25 different property uses - let's compress them down to 4 categories



    df_train['N-PropType'] = df_train.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })







    #polnomials of the variable



    df_train["N-structuretaxvaluedollarcnt-2"] = df_train["structuretaxvaluedollarcnt"] ** 2



    df_train["N-structuretaxvaluedollarcnt-3"] = df_train["structuretaxvaluedollarcnt"] ** 3







    #Average structuretaxvaluedollarcnt by city



    group = df_train.groupby('regionidcity')['structuretaxvaluedollarcnt'].aggregate('mean').to_dict()



    df_train['N-Avg-structuretaxvaluedollarcnt'] = df_train['regionidcity'].map(group)







    #Deviation away from average



    df_train['N-Dev-structuretaxvaluedollarcnt'] = abs((df_train['structuretaxvaluedollarcnt'] - df_train['N-Avg-structuretaxvaluedollarcnt']))/df_train['N-Avg-structuretaxvaluedollarcnt']

    return df_train



print('Loading data...')

properties2017 = pd.read_csv('../input/properties_2017.csv', low_memory = False)

train2016 = pd.read_csv('../input/train_2016_v2.csv')

train2017 = pd.read_csv('../input/train_train_2017.csv')

valid2017 = pd.read_csv('../input/train_valid_2017.csv')



sample_submission = pd.read_csv('../input/sample_submission.csv', low_memory = False)

train2016 = pd.merge(train2016, properties2017, how = 'left', on = 'parcelid')

train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')

valid2017 = pd.merge(valid2017, properties2017, how = 'left', on = 'parcelid')

train2017[['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']] = np.nan

valid2017[['structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount']] = np.nan

train = pd.concat([train2016, train2017], axis = 0)

test = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns = {'parcelid': 'ParcelId'}), 

                how = 'left', on = 'ParcelId')

print('Feature engineering 1...') 

       
'''
train = feature_engineering(train)

test = feature_engineering(test)

valid = feature_engineering(valid2017)
'''
valid = valid2017
del properties2017, train2016, train2017

gc.collect();





print('Memory usage reduction...')

train[['latitude', 'longitude']] /= 1e6

valid[['latitude', 'longitude']] /= 1e6

test[['latitude', 'longitude']] /= 1e6



train['censustractandblock'] /= 1e12

valid['censustractandblock'] /= 1e12

test['censustractandblock'] /= 1e12



        

print('Feature engineering 2...')

train['month'] = (pd.to_datetime(train['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(train['transactiondate']).dt.month

valid['month'] = (pd.to_datetime(valid['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(valid['transactiondate']).dt.month

train = train.drop('transactiondate', axis = 1)

from sklearn.preprocessing import LabelEncoder

non_number_columns = train.dtypes[train.dtypes == object].index.values



for column in non_number_columns:

    train_test = pd.concat([train[column], valid[column]], axis = 0)

    encoder = LabelEncoder().fit(train_test.astype(str))

    train[column] = encoder.transform(train[column].astype(str)).astype(np.int32)

    valid[column] = encoder.transform(valid[column].astype(str)).astype(np.int32)

    
print(train.columns[2:])
feature_names = [feature for feature in train.columns[2:] if feature not in ['parcelid','logerror','month']]
#feature_names = [feature for feature in train.columns[2:] if feature != 'month']


month_avgs = train.groupby('month').agg('mean')['logerror'].values - train['logerror'].mean()

                             

print('Preparing arrays and throwing out outliers...')

X_train = train[feature_names].values

y_train = train['logerror'].values

X_test = valid[feature_names].values

y_test = valid['logerror'].values



del test

gc.collect();



month_values = train['month'].values

month_avg_values = np.array([month_avgs[month - 1] for month in month_values]).reshape(-1, 1)

X_train = np.hstack([X_train, month_avg_values])



X_train = X_train[np.abs(y_train) < 0.4, :]

y_train = y_train[np.abs(y_train) < 0.4]


kfolds = 5

month_values = valid['month'].values

month_avg_values = np.array([month_avgs[18] for month in month_values]).reshape(-1, 1)

X_test = np.hstack([X_test, month_avg_values])



models = []

valid_scores = 0.0

kfold = KFold(n_splits = kfolds, shuffle = True)

from sklearn.metrics import mean_absolute_error
y_preds = None
for i, (train_index, test_index) in enumerate(kfold.split(X_train, y_train)):

    

    print('Training LGBM model with fold {}...'.format(i + 1))

    X_train_, y_train_ = X_train[train_index], y_train[train_index]

    X_valid_, y_valid_ = X_train[test_index], y_train[test_index]

    

    ltrain = lgb.Dataset(X_train_, label = y_train_, free_raw_data = False)

    lvalid = lgb.Dataset(X_valid_, label = y_valid_, free_raw_data = False)

    

    params = {}

    params['metric'] = 'mae'

    params['max_depth'] = 100

    params['num_leaves'] = 32

    params['feature_fraction'] = .85

    params['bagging_fraction'] = .95

    params['bagging_freq'] = 8

    params['learning_rate'] = 0.01

    params['verbosity'] = 0

    

    model = lgb.train(params, ltrain, valid_sets = [ltrain, lvalid], 

            verbose_eval=200, num_boost_round=800)

            

    pred = model.predict(X_test)
    if(i == 0):
        y_preds = pred
    else:
        y_preds += pred
        
    valid_score = mean_absolute_error(y_test, pred)

    print('Valid MAE =', valid_score)

    valid_scores += valid_score

valid_scores /= kfolds 
y_preds /= kfolds  

print('Average valid MAE =', valid_scores)

valid_score = mean_absolute_error(y_test, y_preds)

print('Blending valid MAE =', valid_score)

output = pd.DataFrame({'valid_pred': y_preds})
output.to_csv('valid_lgbm.csv', index=False, float_format = '%.5f')
                  

