import math
import numpy as np # linear algebra
import pandas as pd # linear algebra
from datetime import datetime
macro=pd.read_csv('macro.csv')
timestamp=macro["timestamp"]
cpi=macro["cpi"]
cpi_base=407.0
time_cpi_rate={}
for i,ts in enumerate(timestamp):
	d = datetime.strptime(ts, "%Y/%m/%d")
	time_cpi_rate[d]=math.sqrt(math.sqrt(float(cpi[i])/cpi_base))
	print(d)
fr = open("train.csv", 'r')
fw = open("train_fix_price.csv", 'w')
index=0
header=fr.readline().replace("\n","")
lines=fr.readlines()
fw.writelines(header+"\n")
for line in lines:
	data=line.replace("\n","").split(',')
	ts=data[1]
	d = datetime.strptime(ts, "%Y-%m-%d")
	price_doc=float(data[-1])
	data[-1]=str(price_doc/time_cpi_rate[d])
	fw.writelines(','.join(data)+"\n")
fr.close()
fw.close()
