import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
targets = [201502,201503,201504,201505,201506,201507,201508,201509,201510,201511,201512,
            201601,201602,201603,201604,201605,201606,201607,201608,201609,201610,201611,201612,
            201701,201702,201703]
for target in targets:
    print(target)
    df1 = pd.read_csv('label_all/user_label_{}.csv'.format(target))
    df1['is_churn'] = df1['is_churn'].astype(np.int)
    df1.to_csv('label_all/user_label_fix_{}.csv'.format(target),index=False)