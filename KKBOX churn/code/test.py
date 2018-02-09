import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
df1 = pd.read_csv('train_v2_final_201702.csv')
df1['is_churn_label'] = df1['is_churn'].astype(np.int)
df1 = df1.drop(['is_churn'], axis=1)
df2 = pd.read_csv('train_v2_201702.csv')
msno1 = df1['msno'].values
msno2 = df2['msno'].values

df1 = df1[df1['msno'].isin(msno2)].reset_index(drop=True)
df1 = df1.rename({'is_churn':'is_churn_my'})
df3 = pd.merge(df2,df1,how='left',on='msno')
df3.to_csv('test_labeler.csv',index=False)