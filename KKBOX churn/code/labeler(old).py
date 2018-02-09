import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)
transactions = transactions.sort_values(['msno','transaction_date'])
current_msno = ''

targets = [201601,201602,201603,201604,201605,201606,
            201607,201608,201609,201610,201611,201612,
            201701,201702,201703,201704]
msnos = [[] for i in range(len(targets))]
is_churns = [[] for i in range(len(targets))]
dates = [[] for i in range(len(targets))]
tdates = [[] for i in range(len(targets))]
last_is_churns = [[] for i in range(len(targets))]
churn_rates = [[] for i in range(len(targets))]
transaction_counts = [[] for i in range(len(targets))]
churn_counts = [[] for i in range(len(targets))]
payment_method_ids_all = [[] for i in range(len(targets))]
payment_plan_days_all = [[] for i in range(len(targets))]
plan_list_prices_all = [[] for i in range(len(targets))]
actual_amount_paids_all = [[] for i in range(len(targets))]
is_auto_renews_all = [[] for i in range(len(targets))]

is_cancels = []
transaction_dates = []
membership_expire_dates = []
payment_method_ids = []
payment_plan_days = []
plan_list_prices = []
actual_amount_paids = []
is_auto_renews = []
total_rows = len(transactions['msno'])

msno2churn = {}

for i, row in tqdm(transactions.iterrows(), total=total_rows):
    msno = row['msno']
    transaction_date = row['transaction_date']
    membership_expire_date = row['membership_expire_date']
    is_cancel = int(row['is_cancel'])
    payment_method_id = row['payment_method_id']
    payment_plan_day = row['payment_plan_days']
    plan_list_price = row['plan_list_price']
    actual_amount_paid = row['actual_amount_paid']
    is_auto_renew = row['is_auto_renew']
    if(current_msno != msno or i == total_rows - 1):
        if(current_msno != ''):
            for z, target in enumerate(targets):
                if(z == 0):
                    continue
                for j in range(len(transaction_dates)):
                    l = int(transaction_dates[j]/100)
                    r = int(membership_expire_dates[j]/100)
                    record_index = j
                    if(l == targets[z-1]):
                    #如果交易日在目標月份前一月
                        expired_date = membership_expire_dates[j]
                        for k in range(j+1,len(transaction_dates)):
                            l2 = int(transaction_dates[k]/100)
                            if(l2 == targets[z-1]):
                            #還在目標月份
                                if(transaction_dates[k] > transaction_dates[record_index]):
                                # 如果交易日是之後，則一定依照這筆交易的到期日為最終到期日
                                    expired_date = membership_expire_dates[k]
                                    record_index = k
                                else:
                                # 如果同一天，則會有不同狀況
                                    x_sig = plan_list_prices[record_index] + payment_plan_days[record_index] + payment_method_id[record_index]
                                    y_sig = plan_list_prices[k] + payment_plan_days[k] + payment_method_id[k]
                                    if(x_sig != y_sig):
                                        if(y_sig < x_sig):
                                            expired_date = membership_expire_dates[k]
                                            record_index = k
                                    else:
                                        if(is_cancels[k] == 1 and is_cancels[record_index] == 1):
                                        # multiple cancel, consecutive cancels should only put the expiration date earlier
                                            if(membership_expire_dates[k] < membership_expire_dates[record_index]):
                                                expired_date = membership_expire_dates[k]
                                                record_index = k
                                        elif(is_cancels[k] == 0 and is_cancels[record_index] == 0):
                                        # multiple renewal, expiration date keeps extending
                                            if(membership_expire_dates[k] > membership_expire_dates[record_index]):
                                                expired_date = membership_expire_dates[k]
                                                record_index = k
                                        else:
                                        # same day same plan transaction: subscription preceeds cancellation
                                            if(is_cancels[k] == 1):
                                                expired_date = membership_expire_dates[k]
                                                record_index = k
                                                
                            elif(l2 < targets[z-1]):
                                break
                        
                        if(int(expired_date/100) == target):
                        #如果到期日在目標月份，則是目標candidate，要做label
                            #expired_date = datetime.datetime.strptime(str(expired_date), "%Y%m%d")
                            is_churn = 1
                            #後續日期沒有交易紀錄，則是流失
                            for k in range(j+1,len(transaction_dates)):
                                l2 = int(transaction_dates[k]/100)
                                if(l2 <= targets[z-1]):
                                # 還沒到下個月份
                                    continue
                                gap = 99999
                                trans_date = transaction_dates[k]
                                if(is_cancels[k] == 1):
                                    expire_date2 = membership_expire_dates[k]
                                    if(expired_date > expire_date2):
                                        expired_date = expire_date2
                                for g in range(k+1,len(transaction_dates)):
                                    
                                is_cancel = is_cancels[k]
                                dif_d = (trans_date-expired_date).days
                                if(l2 >= target
                    if(r == target):
                    #如果到期日在目標月份
                        is_churn = 1
                        #後續日期沒有交易紀錄，則是流失
                        expired_date = datetime.datetime.strptime(str(membership_expire_dates[j]), "%Y%m%d")
                        for k in range(j+1,len(transaction_dates)):
                            trans_date = datetime.datetime.strptime(str(transaction_dates[k]), "%Y%m%d")
                            is_cancel = is_cancels[k]
                            dif_d = (trans_date-expired_date).days
                            if(dif_d < 30 and dif_d >= 0):
                                if(is_cancel == 0):
                                    is_churn = 0
                            elif(dif_d >= 30):
                            #因為是照交易日期先後排序，如果超過expired_date 30天，則是流失
                                break
                        msnos[z].append(current_msno) 
                        is_churns[z].append(is_churn)    
                        dates[z].append(membership_expire_dates[j])
                        tdates[z].append(transaction_dates[j])
                        payment_method_ids_all[z].append(payment_method_ids[j])
                        payment_plan_days_all[z].append(payment_plan_days[j])
                        plan_list_prices_all[z].append(plan_list_prices[j])
                        actual_amount_paids_all[z].append(actual_amount_paids[j])
                        is_auto_renews_all[z].append(is_auto_renews[j])
                        last_is_churn = -1
                        churn_rate = -1
                        transaction_count = -1
                        churn_count = -1
                        if(current_msno not in msno2churn):
                            msno2churn[current_msno] = []
                        else:
                            last_is_churn = msno2churn[current_msno][-1]
                            transaction_count = len(msno2churn[current_msno])
                            churn_count = sum(msno2churn[current_msno])
                            churn_rate = churn_count / transaction_count
                        msno2churn[current_msno].append(is_churn)
                        last_is_churns[z].append(last_is_churn)
                        churn_rates[z].append(churn_rate)
                        churn_counts[z].append(churn_count)
                        transaction_counts[z].append(transaction_count)
                        break
            
        transaction_dates = []
        membership_expire_dates = []
        is_cancels = []
        payment_method_ids = []
        payment_plan_days = []
        plan_list_prices = []
        actual_amount_paids = []
        is_auto_renews = []
    transaction_dates.append(transaction_date)
    membership_expire_dates.append(membership_expire_date)
    is_cancels.append(is_cancel)
    payment_method_ids.append(payment_method_id)
    payment_plan_days.append(payment_plan_day)
    plan_list_prices.append(plan_list_price)
    actual_amount_paids.append(actual_amount_paid)
    is_auto_renews.append(is_auto_renew)
    current_msno = msno
for i, target in enumerate(targets):
    out = pd.DataFrame()
    out['msno'] = msnos[i]
    out['is_churn'] = is_churns[i]
    out['expire_date'] = dates[i]
    out['transaction_date'] = tdates[i]
    out['payment_method_id'] = payment_method_ids_all[i]
    out['payment_plan_days'] = payment_plan_days_all[i]
    out['plan_list_price'] = plan_list_prices_all[i]
    out['actual_amount_paid'] = actual_amount_paids_all[i]
    out['is_auto_renew'] = is_auto_renews_all[i]
    out['last_is_churn'] = last_is_churns[i]
    out['churn_rate'] = churn_rates[i]
    out['churn_count'] = churn_counts[i]
    out['transaction_count'] = transaction_counts[i]
    out['discount'] = out['plan_list_price'] - out['actual_amount_paid']
    out['is_discount'] = out.discount.apply(lambda x: 1 if x > 0 else 0)
    out['amt_per_day'] = out['actual_amount_paid'] / out['payment_plan_days']
    
    out.to_csv('train_v2_{}.csv'.format(target), index=False)