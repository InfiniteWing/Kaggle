import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)
transactions = transactions.sort_values(['msno','transaction_date','membership_expire_date'])
current_msno = ''

targets = [201702]
msnos = [[] for i in range(len(targets))]
is_churns = [[] for i in range(len(targets))]

is_cancels = []
transaction_dates = []
membership_expire_dates = []

total_rows = len(transactions['msno'])

for i, row in tqdm(transactions.iterrows(), total=total_rows):
    msno = row['msno']
    transaction_date = row['transaction_date']
    membership_expire_date = row['membership_expire_date']
    is_cancel = int(row['is_cancel'])
    if(current_msno != msno or i == total_rows - 1):
        if(current_msno != ''):
            for z, target in enumerate(targets):
                for j in range(len(transaction_dates)-1,-1,-1):
                    l = int(transaction_dates[j]/100)
                    r = int(membership_expire_dates[j]/100)
                    record_index = j
                    if(r == target and is_cancels[j] == 0 and l < target):
                    #如果到期日在目標月份，不是取消，同時交易日要在目標月以前
                        is_churn = 1
                        #後續日期沒有交易紀錄，則是流失
                        expired_date = datetime.datetime.strptime(str(membership_expire_dates[j]), "%Y%m%d")
                        for k in range(j+1,len(transaction_dates)):
                            trans_date = datetime.datetime.strptime(str(transaction_dates[k]), "%Y%m%d")
                            expired_date1 = datetime.datetime.strptime(str(membership_expire_dates[k]), "%Y%m%d")
                            is_cancel = is_cancels[k]
                            dif_d = (trans_date-expired_date).days
                            if(dif_d < 30):
                                if(is_cancel == 0 and expired_date < expired_date1):
                                    is_churn = 0
                            else:
                            #因為是照交易日期先後排序，如果超過expired_date 30天，則是流失
                                break
                        msnos[z].append(current_msno) 
                        is_churns[z].append(is_churn)   
                        break
            
        transaction_dates = []
        membership_expire_dates = []
        is_cancels = []
    transaction_dates.append(transaction_date)
    membership_expire_dates.append(membership_expire_date)
    is_cancels.append(is_cancel)
    
    current_msno = msno
for i, target in enumerate(targets):
    out = pd.DataFrame()
    out['msno'] = msnos[i]
    out['is_churn'] = is_churns[i]
    out.to_csv('train_v2_final_{}.csv'.format(target), index=False)