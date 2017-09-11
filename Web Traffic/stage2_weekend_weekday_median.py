import pandas as pd
import numpy as np
import statistics
import math
from sklearn import datasets, linear_model
import datetime
import re
train = pd.read_csv("../input/train_2.csv",encoding='utf-8')
test = pd.read_csv("../input/key_2.csv",encoding='utf-8')

def get_language(page):
    page=str(page)
    res = re.search('[a-z][a-z].wikipedia.org',page)
    try:
        if res:
            return res.group(0)[:2]
    except:
        print(page)
    return 'na'

def isweekend(a):
    date_str=a[-10:]
    date=datetime.datetime.strptime(date_str, "%Y-%m-%d")
    if(date.weekday()>=5):
        return True
    return False

def date_index(a):
    date_str=a[-10:]
    date=datetime.datetime.strptime(date_str, "%Y-%m-%d")
    base_date=datetime.datetime(2017,9,1)
    day=(date-base_date).days
    return day


test['is_weekend']=test.Page.apply(isweekend)
test['date_index']=test.Page.apply(date_index)
test['lang']=test.Page.apply(get_language)
test['Page'] = test.Page.apply(lambda a: a[:-11])
print(test.head(10))
visit_weekday_dict={}
visit_weekend_dict={}

headers=list(train.columns.values)
headers_is_weekend=[]
for date_str in headers:
    try:
        date=datetime.datetime.strptime(date_str, "%Y-%m-%d")
        if(date.weekday()>=5):
            headers_is_weekend.append(True)
        else:
            headers_is_weekend.append(False)
    except:
        pass

print(headers_is_weekend[-7:])

def get_avg_without_outliers(a):
    try:
        sd=statistics.stdev(a)
        avg=statistics.mean(a)
        b=[c for c in a if(abs(c-avg)<=1*sd)]
    except:
        b=a
    return statistics.mean(b)
        
for index, row in train.iterrows():
    visits_weekend=[]
    visits_weekday=[]
    chk=0
    for i in range(365):
        visit=row[train.columns[-1*(1+i)]]
        if(np.isnan(visit)):
            continue
        if(headers_is_weekend[-1*(1+i)]):
            visits_weekend.append(visit)
            chk+=1
        else:
            visits_weekday.append(visit)
            chk+=1
        if(chk>60):
            break
    page_str=row['Page']
    if(len(visits_weekend)>0):
        visit_weekend=statistics.median(visits_weekend[:14])
    else:
        if(len(visits_weekday)>0):
            visit_weekend=statistics.median(visits_weekday[:35])
        else:
            visit_weekend=0
    if(len(visits_weekday)>0):
        visit_weekday=statistics.median(visits_weekday[:35])
    else:
        if(len(visits_weekend)>0):
            visit_weekday=statistics.median(visits_weekend[:14])
        else:
            visit_weekday=0
    
    visit_weekday_dict[page_str]=visit_weekday
    visit_weekend_dict[page_str]=visit_weekend
    
visits=[]
for index, row in test.iterrows():
    if(row['is_weekend']):
        visit=visit_weekend_dict[row['Page']]
    else:
        visit=visit_weekday_dict[row['Page']]
    date_index=int(row['date_index'])
    dif_rate = date_index / 61
    if(row['lang']=='en' or row['lang']=='fr'):
        visit_rise_rate=0.012
    else:
        visit_rise_rate=0.008
    visit=int(visit*(1+visit_rise_rate*dif_rate)+0.5)
    visits.append(visit)
test['Visits']=visits
test[['Id','Visits']].to_csv('weekend_weekday_mean_tuned.csv', index=False)


