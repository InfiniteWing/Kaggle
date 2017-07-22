import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pred1=pd.read_csv('CNN_baseline.csv')
pred2=pd.read_csv('My_baseline.csv')

pred_mix={}
for c in pred1.columns:
	if(c!='test_id'):
		pred_mix[c]=[]
		for i,v1 in enumerate(pred1[c]):
			v2=pred2[c][i]
			v3=int(((int(v1)+int(v2))/2)*1.1+0.5)
			pred_mix[c].append(v3)
pred_mix["adult_males"]=np.array(pred_mix["adult_males"])
pred_mix["subadult_males"]=np.array(pred_mix["subadult_males"])
pred_mix["adult_females"]=np.array(pred_mix["adult_females"])*1.35
pred_mix["juveniles"]=np.array(pred_mix["juveniles"])
pred_mix["pups"]=np.array(pred_mix["pups"])*1.65

output = pd.DataFrame({'test_id': pred1['test_id'].astype(np.int32),
		'adult_males': pred_mix["adult_males"].astype(np.int32),
		'subadult_males': pred_mix["subadult_males"].astype(np.int32),
		'adult_females': pred_mix["adult_females"].astype(np.int32),
		'juveniles': pred_mix["juveniles"].astype(np.int32),
		'pups': pred_mix["pups"].astype(np.int32)})
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
from datetime import datetime
output.to_csv('sub_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)