import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
file_ins = {}
for df in df_iter:
    for i, row in tqdm(df.iterrows(),total=len(df)):
        out = []
        out.append(str(row['msno']))
        out.append(str(row['date']))
        out.append(str(row['num_25']))
        out.append(str(row['num_50']))
        out.append(str(row['num_75']))
        out.append(str(row['num_985']))
        out.append(str(row['num_100']))
        out.append(str(row['num_unq']))
        out.append(str(row['total_secs']))
        d = str(int(int(row['date']) / 100))
        if(d not in file_ins):
            fin = open('F:/kkbox/user_logs_{}.csv'.format(d),'w+',encoding='utf8')
            fin.write('msno,date,num_25,num_50,num_75,num_985,num_100,num_unq,total_secs'+'\n')
            file_ins[d] = fin
        file_ins[d].write(','.join(out)+'\n')