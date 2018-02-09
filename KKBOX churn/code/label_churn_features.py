import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
#201704 come from sample_submission_v2
targets = [201502,201503,201504,201505,201506,201507,201508,201509,201510,201511,201512,
            201601,201602,201603,201604,201605,201606,201607,201608,201609,201610,201611,201612,
            201701,201702,201703,201704]
msno2churns = {}
for i,target in enumerate(targets):
    print(target)
    df = pd.read_csv('label_all/user_label_fix_{}.csv'.format(target))
    last_is_churn = []
    churn_rate = []
    churn_count = []
    transaction_count = []
    for j,row in tqdm(df.iterrows(),total=len(df)):
        msno = row['msno']
        is_churn = row['is_churn']
        if(msno not in msno2churns):
            msno2churns[msno] = []
            last_is_churn.append(-1)
            churn_rate.append(-1)
            churn_count.append(-1)
            transaction_count.append(-1)
        else:
            last_is_churn.append(msno2churns[msno][-1])
            churn_rate.append(sum(msno2churns[msno]) / len(msno2churns[msno]))
            churn_count.append(sum(msno2churns[msno]))
            transaction_count.append(len(msno2churns[msno]))
        msno2churns[msno].append(is_churn)
    df['last_is_churn'] = last_is_churn
    df['churn_rate'] = churn_rate
    df['churn_count'] = churn_count
    df['transaction_count'] = transaction_count
    
    df.to_csv('label_all/train_v4_{}.csv'.format(target),index=False)
    
    
    