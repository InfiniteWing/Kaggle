import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)
transactions = transactions.sort_values(['msno','transaction_date'])
current_msno = ''
cols_2015 = ['2015{}'.format(str(i+1).zfill(2)) for i in range(12)]
cols_2016 = ['2016{}'.format(str(i+1).zfill(2)) for i in range(12)]
cols_2017 = ['2017{}'.format(str(i+1).zfill(2)) for i in range(3)]
cols = ['msno'] + cols_2015 + cols_2016 + cols_2017 + ['cancel_count','cutoff_count','transaction_count','combo_days']
cols_dict = {}
for i, v in enumerate(cols):
    cols_dict[v] = i
membership_records = [[] for i in range(len(cols))]
transaction_dates = []
membership_expire_dates = []
is_cancels = []
total_rows = len(transactions['msno'])
for i, row in tqdm(transactions.iterrows(), total=total_rows):
    msno = row['msno']
    transaction_date = row['transaction_date']
    is_cancel = int(row['is_cancel'])
    membership_expire_date = row['membership_expire_date']
    if(current_msno != msno or i == total_rows - 1):
        if(current_msno != ''):
            cutoff_count = 0
            cancel_count = sum(is_cancels)
            transaction_count = len(transaction_dates)
            # membership_record[-1] 是目前連續會員天數
            # membership_record[-2] 是transaction_count
            # membership_record[-3] 是cutoff_count
            # membership_record[0] 是msno
            membership_record = [-1 for j in range(len(cols))]
            left = int(transaction_dates[0])
            right = int(membership_expire_dates[0])
            for j in range(1, len(transaction_dates)):
                l = int(transaction_dates[j])
                r = int(membership_expire_dates[j])
                if(r < right):
                #如果是取消訂閱，新的到期日會比之前的近
                    right = r
                elif(l > right):
                #如果交易日超出到期日，代表有幾天是沒會員
                #同時要先儲存目前的會員狀況
                    cutoff_count += 1
                    start = int(left / 100) #e.g. 20160223 / 100 = 201602
                    end = int(right / 100) #e.g. 20160223 / 100 = 201602
                    for j in range(start, end + 1):
                        if(j % 100 <= 12):
                            if(str(j) in cols_dict):
                                membership_record[cols_dict[str(j)]] = 1
                    left = l
                    right = r
                else:
                #正常到期前續約，延展到期日
                #開始日一樣是一開始的日期
                    right = r
            # 儲存最終的會員狀況        
            start = int(left / 100) #e.g. 20160223 / 100 = 201602
            end = int(right / 100) #e.g. 20160223 / 100 = 201602
            for j in range(start, end + 1):
                if(j % 100 <= 12):
                    if(str(j) in cols_dict):
                        membership_record[cols_dict[str(j)]] = 1
            s_dt = datetime.datetime.strptime(str(left), "%Y%m%d")
            e_dt = datetime.datetime.strptime(str(right), "%Y%m%d")
            membership_record[0] = current_msno
            membership_record[-1] = (e_dt-s_dt).days
            membership_record[-2] = transaction_count
            membership_record[-3] = cutoff_count
            membership_record[-4] = cancel_count
            for j in range(len(cols)):
                membership_records[j].append(membership_record[j])
        transaction_dates = []
        membership_expire_dates = []
        is_cancels = []
        
    transaction_dates.append(transaction_date)
    membership_expire_dates.append(membership_expire_date)
    is_cancels.append(is_cancel)
    current_msno = msno
out = pd.DataFrame()
for i, v in enumerate(cols):
    out[v] = membership_records[i]
out.to_csv('membership_records.csv', index=False)