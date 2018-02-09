import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
targets = [201512,201601,201602,201603,201604,201605,201606,
            201607,201608,201609,201610,201611,201612,
            201701,201702,201703,201704]
members = pd.read_csv('../input/members_v3.csv')
members = members.drop(['city','bd','gender','registered_via'],axis=1)
members_msno = members['msno'].values
for k, target in enumerate(targets):
    if(k == 0):
        continue
    print('Start fe in {}'.format(target))
    train = pd.read_csv('train_{}.csv'.format(targets[k]))
    train = train[train['msno'].isin(members_msno)].reset_index(drop=True)
    train = pd.merge(train, members, how='left', on='msno')
    
    train['registration_year'] = train['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
    train['registration_month'] = train['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
    train['registration_date'] = train['registration_init_time'].apply(lambda x: int(str(x)[6:8]))
 
    for col in ['registration_init_time', 'expire_date', 'transaction_date']:
        train[col] = pd.to_datetime(train[col], format='%Y%m%d')
    train['registration_expired_duration'] = (train['expire_date'] - train['registration_init_time']).dt.days.astype(int)
    train['membership_days'] = (train['transaction_date'] - train['registration_init_time']).dt.days.astype(int)
    

    train = train.drop(['registration_init_time','expire_date','transaction_date'],axis=1)
    train_msnos = train['msno'].values
    print(train.shape)
    #user_logs_1 = pd.read_csv('F:/kkbox/user_logs_{}.csv'.format(targets[k-1]))
    #user_logs_2 = pd.read_csv('F:/kkbox/user_logs_{}.csv'.format(targets[k]))
    #user_logs = user_logs_1.append(user_logs_2).reset_index(drop=True)
    user_logs = pd.read_csv('F:/kkbox/user_logs_{}.csv'.format(targets[k-1]))
    user_logs = user_logs.drop(['date'],axis=1)
    print(user_logs.shape)
    user_logs = user_logs[user_logs['msno'].isin(train_msnos)].reset_index(drop=True)
    print(user_logs.shape)
    user_logs_mean = user_logs.groupby(['msno']).mean().reset_index()
    columns = list(user_logs_mean.columns)
    for i,v in enumerate(columns):
        if(v != 'msno'):
            columns[i] = v + '_mean'
    user_logs_mean.columns = columns
    
    user_logs_sum = user_logs.groupby(['msno']).sum().reset_index()
    columns = list(user_logs_sum.columns)
    for i,v in enumerate(columns):
        if(v != 'msno'):
            columns[i] = v + '_sum'
    user_logs_sum.columns = columns
    
    user_logs_count = user_logs.groupby(['msno']).size().reset_index(name='count')
    print('mean = {}, sum = {}, count = {}'.format(user_logs_mean.shape,user_logs_sum.shape,user_logs_count.shape))
    user_logs = pd.merge(user_logs_mean, user_logs_sum, how='left', on='msno')
    user_logs = pd.merge(user_logs, user_logs_count, how='left', on='msno')
    print(user_logs.shape)
    train_new = pd.merge(train, user_logs, how='left', on='msno')
    print(train_new.shape)
    train_new.to_csv('train_pre_{}.csv'.format(target),index=False)