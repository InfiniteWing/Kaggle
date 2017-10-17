
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
valid = pd.read_csv('../input/train_valid_2017.csv')
y_valid = valid['logerror'].values
m1 = pd.read_csv('valid_lgbm.csv')
m2 = pd.read_csv('valid_xgb.csv')
m3 = pd.read_csv('valid_catboost.csv')
m4 = pd.read_csv('valid_GP.csv')

valid_score = mean_absolute_error(y_valid, m1['valid_pred'])
print('LGBM valid MAE =', valid_score)

valid_score = mean_absolute_error(y_valid, m2['valid_pred'])
print('XGB valid MAE =', valid_score)

valid_score = mean_absolute_error(y_valid, m3['valid_pred'])
print('CatBoost valid MAE =', valid_score)

valid_score = mean_absolute_error(y_valid, m4['valid_pred'])
print('GP valid MAE =', valid_score)

use_models = [m1, m2, m3, m4]
weights = [0.29, 0.28, 0.3, 0.13]
use_cols = ['valid_pred']
ensemble_method = 'mean'
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

valid_score = mean_absolute_error(y_valid, ensemble_preds['valid_pred'])

print('Ensembling valid MAE =', valid_score)



