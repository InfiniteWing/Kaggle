
# Original kernel: https://www.kaggle.com/rshally/web-traffic-cross-valid-round-and-wk-lb-44-5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
Windows = [14,31,49,90,120,180,365]
Windows=np.array(Windows)

train = pd.read_csv("../input/train_2.csv",encoding='utf-8')
train.fillna(0, inplace=True)
test = pd.read_csv('../input/key_2.csv',encoding='utf-8')
test['Date'] = test.Page.apply(lambda x: x[-10:])
test['Page'] = test.Page.apply(lambda x: x[:-11])
test['Date'] = test['Date'].astype('datetime64[ns]')
test['wk']= test.Date.dt.dayofweek >=5

for i in Windows:
    print(i,end= ' ')
    val='MW'+str(i)
    tmp = pd.melt(train[list(train.columns[-i:])+['Page']], 
                  id_vars='Page', var_name='D', value_name=val)
    tmp['D'] = tmp['D'].astype('datetime64[ns]')
    tmp['wk']= tmp.D.dt.dayofweek  >=5           
    tmp1 = tmp.groupby(['Page','wk']).median().reset_index()
    test = test.merge(tmp1, how='left')
    
test['Visits']=test.iloc[:,4:].median(axis=1).round().astype(int)
test[['Id','Visits']].to_csv('submission_Rudolph_solution.csv', index=False)
gc.collect()