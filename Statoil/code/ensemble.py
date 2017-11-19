

import numpy as np
import pandas as pd
import xgboost as xgb

m1 = pd.read_csv("../input/0.193.csv")
m2 = pd.read_csv("../input/0.203.csv")
m3 = pd.read_csv("../input/0.210.csv")
use_models = [m1, m2, m3]
weights = [0.4, 0.33, 0.27]
use_cols = ['is_iceberg']
ensemble_method = 'mean'# 'mean' or 'median'
ensemble_preds = {}
for col in use_cols:
    preds = None
    if(ensemble_method == 'mean'): 
        preds = None
        for i,model in enumerate(use_models):
            if(preds is None):
                preds = model[col] * weights[i]
            else:
                preds += model[col] * weights[i]
        #preds /= len(use_models)
    else:
        combined = np.vstack((model[col] for model in use_models))
        preds = np.median(combined, axis = 0)
    ensemble_preds[col] = preds


output = pd.DataFrame({'id': use_models[0]['id']})
for col in use_cols:
    output[col] =  ensemble_preds[col]
   
output.to_csv('ensemble.csv', index=False, float_format = '%.8f')


