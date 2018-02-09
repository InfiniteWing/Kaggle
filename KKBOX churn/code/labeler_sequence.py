import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

targets = [201601,201602,201603,201604,201605,201606,
            201607,201608,201609,201610,201611,201612,
            201701,201702,201703]
df_all = pd.read_csv('train_201703.csv')
for i in range(len(targets)-2,-1,-1):
    print(i)
    df = pd.read_csv('train_{}.csv'.format(targets[i]))
    df_all = pd.concat([df_all, df]).reset_index(drop=True)
    df_all = df_all.drop_duplicates(subset=['msno'], keep='first')
df_all = df_all.drop(['year_month'], axis=1)
df_all.to_csv('train_all.csv', index=False)