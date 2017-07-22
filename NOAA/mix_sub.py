import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
tests=[]
for i in range(6):
    test = pd.read_csv("sub_24_{}.csv".format(i))
    tests.append(test)
test=pd.concat(tests)
test.to_csv('sub_24.csv', index=False)