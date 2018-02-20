import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import gc

K = 4
df_train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
print('Shape train: {}\nShape test: {}'.format(df_train.shape,df_test.shape))

median_price = df_train['price'].median()
mean_price = df_train['price'].mean()
exclude_cols = ['name', 'category_name', 'brand_name', 'item_description', 'category_1', 'category_2', 'category_3']

# start https://www.kaggle.com/golubev/naive-xgboost-v2
def change_datatype(df):
    for col in list(df.select_dtypes(include=['int']).columns):
        if np.max(df[col]) <= 127 and np.min(df[col]) >= -128:
            df[col] = df[col].astype(np.int8)
        elif np.max(df[col]) <= 255 and np.min(df[col]) >= 0:
            df[col] = df[col].astype(np.uint8)
        elif np.max(df[col]) <= 32767 and np.min(df[col]) >= -32768:
            df[col] = df[col].astype(np.int16)
        elif np.max(df[col]) <= 65535 and np.min(df[col]) >= 0:
            df[col] = df[col].astype(np.uint16)
        elif np.max(df[col]) <= 2147483647 and np.min(df[col]) >= -2147483648:
            df[col] = df[col].astype(np.int32)
        elif np.max(df[col]) <= 4294967296 and np.min(df[col]) >= 0:
            df[col] = df[col].astype(np.uint32)
    for col in list(df.select_dtypes(include=['float']).columns):
        df[col] = df[col].astype(np.float32)
# end https://www.kaggle.com/golubev/naive-xgboost-v2
def category_detail(x, i):
    try:
        x = x.split('/')[i]
        return x
    except:
        return ''
def category_detail_1(x):
    return category_detail(x, 0)
def category_detail_2(x):
    return category_detail(x, 1)
def category_detail_3(x):
    return category_detail(x, 2)

def median_price_features(df_train, df_test, col):
    price_dict = df_train.groupby(col)['price'].median().to_dict()
    tmp = pd.DataFrame({
        col:list(price_dict.keys()),
        '{}_median_price'.format(col):list(price_dict.values())})
    
    df_train = pd.merge(df_train, tmp, how='left', on=col)
    df_train['{}_median_price'.format(col)].fillna(median_price, inplace=True)
    df_train['{}_median_price'.format(col)] = df_train['{}_median_price'.format(col)].astype(np.int16)
    
    df_test = pd.merge(df_test, tmp, how='left', on=col)
    df_test['{}_median_price'.format(col)].fillna(median_price, inplace=True)
    df_test['{}_median_price'.format(col)] = df_test['{}_median_price'.format(col)].astype(np.int16)
    
    return df_train, df_test
    
def mean_price_features(df_train, df_test, col):
    price_dict = df_train.groupby(col)['price'].mean().to_dict()
    tmp = pd.DataFrame({
        col:list(price_dict.keys()),
        '{}_mean_price'.format(col):list(price_dict.values())})
    
    df_train = pd.merge(df_train, tmp, how='left', on=col)
    df_train['{}_mean_price'.format(col)].fillna(mean_price, inplace=True)
    df_train['{}_mean_price'.format(col)] = df_train['{}_mean_price'.format(col)].astype(np.int16)
    
    df_test = pd.merge(df_test, tmp, how='left', on=col)
    df_test['{}_mean_price'.format(col)].fillna(mean_price, inplace=True)
    df_test['{}_mean_price'.format(col)] = df_test['{}_mean_price'.format(col)].astype(np.int16)
    return df_train, df_test
    
change_datatype(df_train)
change_datatype(df_test)
print('Shape train: {}\nShape test: {}'.format(df_train.shape,df_test.shape))
df_train['category_1'] = df_train['category_name'].apply(category_detail_1)
df_train['category_2'] = df_train['category_name'].apply(category_detail_2)
df_train['category_3'] = df_train['category_name'].apply(category_detail_3)

df_test['category_1'] = df_test['category_name'].apply(category_detail_1)
df_test['category_2'] = df_test['category_name'].apply(category_detail_2)
df_test['category_3'] = df_test['category_name'].apply(category_detail_3)
print('Shape train: {}\nShape test: {}'.format(df_train.shape,df_test.shape))

df_train, df_test = median_price_features(df_train, df_test, 'category_name')
df_train, df_test = median_price_features(df_train, df_test, 'brand_name')
df_train, df_test = median_price_features(df_train, df_test, 'category_1')
df_train, df_test = median_price_features(df_train, df_test, 'category_2')
df_train, df_test = median_price_features(df_train, df_test, 'category_3')

change_datatype(df_train)
change_datatype(df_test)

df_train, df_test = mean_price_features(df_train, df_test, 'category_name')
df_train, df_test = mean_price_features(df_train, df_test, 'brand_name')
df_train, df_test = mean_price_features(df_train, df_test, 'category_1')
df_train, df_test = mean_price_features(df_train, df_test, 'category_2')
df_train, df_test = mean_price_features(df_train, df_test, 'category_3')

change_datatype(df_train)
change_datatype(df_test)

print('Shape train: {}\nShape test: {}'.format(df_train.shape,df_test.shape))

# start https://www.kaggle.com/golubev/naive-xgboost-v2
c_texts = ['name', 'item_description']
def count_words(key):
    return len(str(key).split())
def count_numbers(key):
    return sum(c.isalpha() for c in str(key))
def count_upper(key):
    return sum(c.isupper() for c in str(key))
for c in c_texts:
    df_train[c + '_c_words'] = df_train[c].apply(count_words)
    df_train[c + '_c_upper'] = df_train[c].apply(count_upper)
    df_train[c + '_c_numbers'] = df_train[c].apply(count_numbers)
    df_train[c + '_len'] = df_train[c].str.len()
    df_train[c + '_mean_len_words'] = df_train[c + '_len'] / df_train[c + '_c_words']
    df_train[c + '_mean_upper'] = df_train[c + '_len'] / df_train[c + '_c_upper']
    df_train[c + '_mean_numbers'] = df_train[c + '_len'] / df_train[c + '_c_numbers']
change_datatype(df_train)

for c in c_texts:
    df_test[c + '_c_words'] = df_test[c].apply(count_words)
    df_test[c + '_c_upper'] = df_test[c].apply(count_upper)
    df_test[c + '_c_numbers'] = df_test[c].apply(count_numbers)
    df_test[c + '_len'] = df_test[c].str.len()
    df_test[c + '_mean_len_words'] = df_test[c + '_len'] / df_test[c + '_c_words']
    df_test[c + '_mean_upper'] = df_test[c + '_len'] / df_test[c + '_c_upper']
    df_test[c + '_mean_numbers'] = df_test[c + '_len'] / df_test[c + '_c_numbers']
change_datatype(df_test)
# end https://www.kaggle.com/golubev/naive-xgboost-v2


change_datatype(df_train)
change_datatype(df_test)
gc.collect()
target_train = df_train['price'].values
train = np.array(df_train['train_id'])
df_train = df_train.drop(['train_id', 'price']+exclude_cols, axis=1)

kf = KFold(n_splits=K, random_state=3228, shuffle=True)

xgb_preds = []
labels = []
scores = 0.0
fold = 1
moedels = []
xgb_params = {
    'eta': 0.08,
    'max_depth': 10,
    'subsample': 0.8,
    'objective': 'reg:linear',
    'silent': 1,
    'eval_metric': 'rmse',
    'colsample_bytree': 0.8
}

for train_index, test_index in kf.split(train): 
    print('Start fold {}'.format(fold))
    train_X, valid_X = df_train.iloc[train_index], df_train.iloc[test_index]
    train_y, valid_y = target_train[train_index], target_train[test_index]    
    print(train_X.columns)
    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    # xgboost, cross-validation
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train, 160, watchlist, early_stopping_rounds=20, verbose_eval=10)
    moedels.append(model)
    fold += 1
del target_train, train
gc.collect()
    
     
for model in moedels:
    print(train_X.columns)
    d_test = xgb.DMatrix(df_test.drop(['test_id']+exclude_cols, axis=1))
    xgb_pred = model.predict(d_test)
    xgb_preds.append(list(xgb_pred))
preds=[]
for i in range(len(xgb_preds[0])):
    sum=0
    for j in range(len(xgb_preds)):
        try:
            sum+=xgb_preds[j][i]
        except:
            print(i,j)
    if(sum > 0):
        preds.append(sum / len(xgb_preds))
    else:
        preds.append(0.1)
        

output = pd.DataFrame({'price': preds, 'test_id': df_test['test_id']})
output = output[['test_id', 'price']]
output.to_csv('sub.csv', index=False)   