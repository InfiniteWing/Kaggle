import numpy as np
import pandas as pd
m1 = pd.read_csv("ensemble_sub_v2.csv")
m2 = pd.read_csv("ensemble_sub_v4.csv")

m1['is_churn'] = (m1['is_churn'] + m2['is_churn']) / 2
m1.to_csv('ensemble_sub_v2v4.csv',index=False)


