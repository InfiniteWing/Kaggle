import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
targets = [201701,201702,201703]
train = pd.read_csv('train_pre_v2_{}.csv'.format(targets[0]))      
for i, target in enumerate(targets):
    if(i == 0):
        continue
    print(train.shape)
    train_2 = pd.read_csv('train_pre_v2_{}.csv'.format(target))
    train = train.append(train_2).reset_index(drop=True)
train.to_csv('train_pre_2017_all_v2.csv',index=False,float_format='%.4f')