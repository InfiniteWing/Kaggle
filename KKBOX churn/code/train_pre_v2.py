import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
targets = [201601,201602,201603,201604,201605,201606,
            201607,201608,201609,201610,201611,201612,
            201701,201702,201703,201704]
for k, target in enumerate(targets):
    if(k < 1):
        continue
    print('Start fe in {}'.format(target))
    train = pd.read_csv('train_v2_{}.csv'.format(targets[k]))
    train_msnos = train['msno'].values
    print(train.shape)
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
    train_new.to_csv('train_pre_v2_{}.csv'.format(target),index=False)
    print(train_new.shape)
    
    '''
    user_logs = pd.read_csv('F:/kkbox/user_logs_{}.csv'.format(targets[k-2]))
    user_logs = user_logs.drop(['date'],axis=1)
    print(user_logs.shape)
    user_logs = user_logs[user_logs['msno'].isin(train_msnos)].reset_index(drop=True)
    print(user_logs.shape)
    user_logs_mean = user_logs.groupby(['msno']).mean().reset_index()
    columns = list(user_logs_mean.columns)
    for i,v in enumerate(columns):
        if(v != 'msno'):
            columns[i] = v + '_mean_v2'
    user_logs_mean.columns = columns
    
    user_logs_sum = user_logs.groupby(['msno']).sum().reset_index()
    columns = list(user_logs_sum.columns)
    for i,v in enumerate(columns):
        if(v != 'msno'):
            columns[i] = v + '_sum_v2'
    user_logs_sum.columns = columns
    
    user_logs_count = user_logs.groupby(['msno']).size().reset_index(name='count_v2')
    print('mean = {}, sum = {}, count = {}'.format(user_logs_mean.shape,user_logs_sum.shape,user_logs_count.shape))
    user_logs = pd.merge(user_logs_mean, user_logs_sum, how='left', on='msno')
    user_logs = pd.merge(user_logs, user_logs_count, how='left', on='msno')
    print(user_logs.shape)
    train_new = pd.merge(train_new, user_logs, how='left', on='msno')
    train_new.to_csv('train_pre_v2_v3_{}.csv'.format(target),index=False)
    '''