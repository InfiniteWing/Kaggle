import numpy as np
import pandas as pd

transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)
transactions = transactions.sort_values(['msno','transaction_date'])
current_msno = ''
for i, row in transactions.iterrows():
    msno = row['msno']
    transaction_date = str(row['transaction_date'])
    membership_expire_date = str(row['membership_expire_date'])
    if(current_msno != msno):
        if(current_msno != ''):
            print(current_msno)
            logs = [transaction_dates[i] + ' to ' + membership_expire_dates[i] for i in range(len(membership_expire_dates))]
            print(' '.join(logs))
            input()
        transaction_dates = []
        membership_expire_dates = []
        
    transaction_dates.append(transaction_date)
    membership_expire_dates.append(membership_expire_date)
    current_msno = msno
    