import numpy as np
import pandas as pd
import csv
df_train = pd.read_csv('../input/train.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
df_test = pd.read_csv('../input/test.tsv', sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)

median = df_train['price'].median()
train = df_train.groupby('category_name')['price'].median()
price_dict = train.to_dict()

preds = []
for i, row in df_test.iterrows():
    category_name = row['category_name']
    if(category_name not in price_dict):
        preds.append(median)
    else:
        preds.append(price_dict[category_name])
        
sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['price'] = preds
sub.to_csv('sub.csv', index=False)