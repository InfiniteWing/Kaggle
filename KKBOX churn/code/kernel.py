import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn

#train = pd.read_csv('../input/train_all.csv')
train = pd.read_csv('../input/train_201703.csv')
#train = pd.concat((train, pd.read_csv('../input/train_201703.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('../input/sample_submission_v2.csv')

transactions = pd.read_csv('../input/transactions.csv', usecols=['msno'])
transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
transactions.columns = ['msno','trans_count']
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions = []; print('transaction merge...')

user_logs = pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])
#user_logs = pd.read_csv('../input/user_logs.csv', usecols=['msno'])
#user_logs = pd.concat((user_logs, pd.read_csv('../input/user_logs_v2.csv', usecols=['msno'])), axis=0, ignore_index=True).reset_index(drop=True)
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']
train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')
user_logs = []; print('user logs merge...')

members = pd.read_csv('../input/members_v3.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = []; print('members merge...') 

gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

train = train.fillna(0)
test = test.fillna(0)

transactions = pd.read_csv('../input/transactions_v2.csv') #pd.read_csv('../input/transactions.csv')
#transactions = pd.concat((transactions, pd.read_csv('../input/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions=[]

def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35:
        if len(df)>0:
            print(df.shape)
            df = transform_df(df) 
            df = transform_df2(df)
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1
last_user_logs.append(transform_df(pd.read_csv('../input/user_logs_v2.csv')))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)

members_features = pd.read_csv('membership_records.csv')

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
def last_play_date(row):
    row[x]
#train['last_play_date'] = train.apply(last_play_date)
#train = pd.merge(train, members_features, how='left', on='msno')
#test = pd.merge(test, members_features, how='left', on='msno')
last_user_logs=[]

train = train.fillna(0)
test = test.fillna(0)

cols = [c for c in train.columns if c not in ['is_churn','msno','transaction_count']]



def xgb_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'log_loss', metrics.log_loss(labels, preds)

fold = 5
for i in range(fold):
    params = {
        'eta': 0.02, #use 0.002
        'max_depth': 6,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': i,
        'silent': True
    }
    x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.2, random_state=i)
    watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
    model = xgb.train(params, xgb.DMatrix(x1, y1), 200,  watchlist, feval=xgb_score, maximize=False, verbose_eval=10, early_stopping_rounds=20) #use 1500
    if i != 0:
        pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
    else:
        pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
pred /= fold
test['is_churn'] = pred.clip(0.+1e-15, 1-1e-15)
test[['msno','is_churn']].to_csv('5fold_sub.csv', index=False)
#test[['msno','is_churn']].to_csv('submission.csv.gz', index=False, compression='gzip')