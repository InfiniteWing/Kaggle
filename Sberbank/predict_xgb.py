import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing
import math
from datetime import datetime

from datetime import datetime
macro=pd.read_csv('macro.csv')
timestamp=macro["timestamp"]
cpi=macro["cpi"]
cpi_base=407.0
time_cpi_rate={}
for i,ts in enumerate(timestamp):
	d = datetime.strptime(ts, "%Y/%m/%d")
	time_cpi_rate[d]=math.sqrt(math.sqrt(float(cpi[i])/cpi_base))

train  = pd.read_csv("train_fix_price.csv")
test = pd.read_csv("test_fix.csv")
ylog_train_all = np.log1p(train ['price_doc'].values)
id_test = test['id']
test_timestamp=test['timestamp']

train.drop(['id','timestamp', 'price_doc'], axis=1, inplace=True)
test.drop(['id','timestamp'], axis=1, inplace=True)

for c in train.columns:
	if train[c].dtype == 'object':
		lbl = preprocessing.LabelEncoder() 
		lbl.fit(list(train[c].values) + list(test[c].values)) 
		train[c] = lbl.transform(list(train[c].values))
		test[c] = lbl.transform(list(test[c].values))
xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train, ylog_train_all)
dtest = xgb.DMatrix(test)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,verbose_eval=10, show_stdv=False)
num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)
ylog_pred = model.predict(dtest)
preds =np.exp(ylog_pred) 
y_pred=[]
for i,pred in enumerate(preds):
	d = datetime.strptime(test_timestamp[i], "%Y-%m-%d")
	cpi_rate=time_cpi_rate[d]
	pred = pred*cpi_rate
	y_pred.append(pred)
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
df_sub.to_csv('sub.csv', index=False)











