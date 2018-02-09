import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
targets = [201601,201602,201603,201604,201605,201606,
            201607,201608,201609,201610,201611,201612,
            201701,201702,201703,201704]
targets = [201702,201703]
for k, target in enumerate(targets):
    if(k < 1):
        continue
    print('Start fe in {}'.format(target))
    train = pd.read_csv('train_pre_v5_{}.csv'.format(targets[k]))
    #print(train.head())
    train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) & (train.is_cancel == 0)).astype(np.int8)
    train.to_csv('train_pre_v5_fix_{}.csv'.format(targets[k]),index=False)
    #print(train.head())
    