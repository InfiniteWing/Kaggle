import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc

df_train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)

print('Load data complete\nShape train: {}\nShape test: {}'.format(df_train.shape,df_test.shape))
median_price = df_train['price'].median()
mean_price = df_train['price'].mean()
exclude_cols = ['name', 'category_name', 'brand_name', 'item_description', 'category_1', 'category_2', 'category_3']

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

def category_features(df_train, df_test, col):
    cate_train = set(df_train[col].unique())
    cate_test = set(df_test[col].unique())
    cate_all = cate_train.union(cate_test)
    print('category {} in train have {} unique values'.format(col, len(cate_train)))
    print('category {} in test have {} unique values'.format(col, len(cate_test)))
    print('category {} in train ∪ test have {} unique values'.format(col, len(cate_all)))
    print()
    tmp = pd.DataFrame({
        col:list(cate_all),
        '{}_cat'.format(col):[i+1 for i in range(len(cate_all))]})
    df_train = pd.merge(df_train, tmp, how='left', on=col) 
    df_train['{}_cat'.format(col)].fillna(-1, inplace=True) 
    df_test = pd.merge(df_test, tmp, how='left', on=col)
    df_test['{}_cat'.format(col)].fillna(-1, inplace=True)
    return df_train, df_test
df_train['category_1'] = df_train['category_name'].apply(category_detail_1)
df_train['category_2'] = df_train['category_name'].apply(category_detail_2)
df_train['category_3'] = df_train['category_name'].apply(category_detail_3)

df_test['category_1'] = df_test['category_name'].apply(category_detail_1)
df_test['category_2'] = df_test['category_name'].apply(category_detail_2)
df_test['category_3'] = df_test['category_name'].apply(category_detail_3)

df_train, df_test = category_features(df_train, df_test, 'category_name')
df_train, df_test = category_features(df_train, df_test, 'brand_name')
df_train, df_test = category_features(df_train, df_test, 'category_1')
df_train, df_test = category_features(df_train, df_test, 'category_2')
df_train, df_test = category_features(df_train, df_test, 'category_3')

df_train, df_test = median_price_features(df_train, df_test, 'category_name')
df_train, df_test = median_price_features(df_train, df_test, 'brand_name')
df_train, df_test = median_price_features(df_train, df_test, 'category_1')
df_train, df_test = median_price_features(df_train, df_test, 'category_2')
df_train, df_test = median_price_features(df_train, df_test, 'category_3')

df_train, df_test = mean_price_features(df_train, df_test, 'category_name')
df_train, df_test = mean_price_features(df_train, df_test, 'brand_name')
df_train, df_test = mean_price_features(df_train, df_test, 'category_1')
df_train, df_test = mean_price_features(df_train, df_test, 'category_2')
df_train, df_test = mean_price_features(df_train, df_test, 'category_3')



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

for c in c_texts:
    df_test[c + '_c_words'] = df_test[c].apply(count_words)
    df_test[c + '_c_upper'] = df_test[c].apply(count_upper)
    df_test[c + '_c_numbers'] = df_test[c].apply(count_numbers)
    df_test[c + '_len'] = df_test[c].str.len()
    df_test[c + '_mean_len_words'] = df_test[c + '_len'] / df_test[c + '_c_words']
    df_test[c + '_mean_upper'] = df_test[c + '_len'] / df_test[c + '_c_upper']
    df_test[c + '_mean_numbers'] = df_test[c + '_len'] / df_test[c + '_c_numbers']
df_train['fe001'] = np.square(df_train["name_mean_len_words"])
df_train['fe002'] = np.square(df_train["item_description_mean_len_words"])
df_train['fe003'] = np.tanh(df_train["name_mean_len_words"])
df_train['fe004'] = np.tanh(df_train["item_description_mean_len_words"])
df_train['fe005'] = df_train["name_mean_len_words"]**2.37
df_train['fe006'] = df_train["item_description_mean_len_words"]**2.15

df_test['fe001'] = np.square(df_test["name_mean_len_words"])
df_test['fe002'] = np.square(df_test["item_description_mean_len_words"])
df_test['fe003'] = np.tanh(df_test["name_mean_len_words"])
df_test['fe004'] = np.tanh(df_test["item_description_mean_len_words"])
df_test['fe005'] = df_test["name_mean_len_words"]**2.37
df_test['fe006'] = df_test["item_description_mean_len_words"]**2.15    
    
# end https://www.kaggle.com/golubev/naive-xgboost-v2



print('After adding features\nShape train: {}\nShape test: {}'.format(df_train.shape,df_test.shape))

gc.collect()
target_train = df_train['price'].values
train = np.array(df_train['train_id'])
df_train = df_train.drop(['train_id', 'price']+exclude_cols, axis=1)

cat_features = []
for i, c in enumerate(df_train.columns):
    if('cat' in c):
        cat_features.append(c)



params = {
    'learning_rate': 0.1,
    'application': 'regression',
    'max_depth': 8,
    'num_leaves': 256,
    'verbosity': -1,
    'metric': 'RMSE'
}
moedels = []
for i in range(5):

    train_X, valid_X, train_y, valid_y = train_test_split(df_train, target_train, test_size = 0.2, random_state = i) 
    d_train = lgb.Dataset(train_X, label=train_y, max_bin=8192)
    d_valid = lgb.Dataset(valid_X, label=valid_y, max_bin=8192)
    watchlist = [d_train, d_valid]

    model = lgb.train(params, train_set=d_train, num_boost_round=300, valid_sets=watchlist, \
    early_stopping_rounds=30, verbose_eval=10, categorical_feature=cat_features) 
    moedels.append(model)

del target_train, train
gc.collect()
    
preds = None
for i, model in enumerate(moedels):
    pred = model.predict(df_test.drop(['test_id']+exclude_cols, axis=1))
    if(i == 0):
        preds = pred
    else:
        preds += pred
preds /= len(moedels)
preds[preds < 0] = 0.01    

output = pd.DataFrame({'price': preds, 'test_id': df_test['test_id']})
output = output[['test_id', 'price']]
output.to_csv('sub.csv', index=False)   