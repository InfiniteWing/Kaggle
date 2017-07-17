import pandas as pd
import numpy as np
import statistics

train = pd.read_csv("../input/train_1.csv")
test = pd.read_csv("../input/key_1.csv")

test['Page'] = test.Page.apply(lambda a: a[:-11])

visits=[]
for index, row in train.iterrows():
    for i in range(8):
        visit=row[train.columns[-49*(i+1):]].median(axis=0, skipna=True)
         
        if(np.isnan(visit)):
            print(index,i)
            continue
        else:
            break
    visits.append(visit)
train['Visits'] = visits
'''
train['Visits'] = train[train.columns[-14:]].median(axis=1, skipna=True)*0.9
'''
test = test.merge(train[['Page','Visits']], how='left')
test.loc[test.Visits.isnull(), 'Visits'] = 0

test[['Id','Visits']].to_csv('med.csv', index=False)