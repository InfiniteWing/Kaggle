#
# m1+m2+m3, mean                = LB 0.0642444
# m1+m2+m3, median              = LB
# m5*0.5 + m6*0.3 + m7*0.2      = LB 0.0641796
# m5*0.6 + m6*0.24 + m7*0.16    = LB 0.0641835
# m5*0.45 + m6*0.33 + m7*0.22   = LB 0.0641795
# m5*0.5 + m10*0.3 + m7*0.2      = LB 0.06415
# m5*0.35 + m10*0.33 + m7*0.32      = LB 0.06415
# m5*0.5 + m11*0.3 + m7*0.2      = LB 0.0641545
# m5*0.5 + m12*0.3 + m7*0.2      = LB 0.0641445
# m13*0.5 + m12*0.3 + m7*0.2      = LB 0.06413
# m14*0.5 + m12*0.3 + m7*0.2      = LB 0.06406
# m15*0.5 + m12*0.3 + m7*0.2      = LB 0.06407
# m15*0.55 + m12*0.45             = LB 0.06407

import numpy as np
import pandas as pd
import xgboost as xgb
#m1 = pd.read_csv("../best/0.06435.csv") # https://www.kaggle.com/seesee/concise-catboost-starter-ensemble-plb-0-06435/output
#m2 = pd.read_csv("../best/0.06437.csv") # https://www.kaggle.com/davidfumo/boosted-trees-lb-0-0643707
#m3 = pd.read_csv("../best/0.06439.csv") # https://www.kaggle.com/scirpus/genetic-programming-lb-0-0643904
#m4 = pd.read_csv("../best/0.06440.csv") # https://www.kaggle.com/vber852/simple-lgbm-model-lb-0-064404/code
'''
m5 = pd.read_csv("../best/0.06425(concise 2017).csv")
m6 = pd.read_csv("../best/0.06435(LGBM).csv")
m7 = pd.read_csv("../best/0.06439(GP).csv")
m8 = pd.read_csv("../best/0.06446(XGBMY).csv")
m9 = pd.read_csv("../best/0.06424(concise 2017).csv")
m10 = pd.read_csv("../best/0.06433(LGBM).csv")
m11 = pd.read_csv("../best/0.06433(LGBM2017).csv")
m12 = pd.read_csv("../best/0.06431(LGBM).csv")
m13 = pd.read_csv("../best/0.06424(concise no outlier).csv")
m14 = pd.read_csv("../best/0.06419(concise 2017).csv")
m15 = pd.read_csv("../best/CatBoost_ver3_by_month.csv")
'''
m16 = pd.read_csv("../best/0.06433(LGBM2017).csv")
m17 = pd.read_csv("../best/sub_xgb_5fold.csv")
m18 = pd.read_csv("../best/CatBoost_ver3_by_month.csv")
m19 = pd.read_csv("../best/0.06439(GP).csv")
#use_models = [m1, m2, m3, m4]
use_models = [m16, m17, m18, m19]
weights = [0.29, 0.28, 0.3, 0.13]
use_cols = ['201610', '201611', '201612', '201710', '201711', '201712']
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


output = pd.DataFrame({'ParcelId': use_models[0]['ParcelId']})
for col in use_cols:
    output[col] =  ensemble_preds[col]
   
output.to_csv('ensemble_xgb_lgb_cat_gp.csv', index=False, float_format = '%.6f')


