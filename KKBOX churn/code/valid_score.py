import gc; gc.enable()
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn import *
import sklearn
from catboost import CatBoostClassifier

pred_1 = pd.read_csv('stack_mean.csv')['is_churn'].values
pred_2 = pd.read_csv('stack_median.csv')['is_churn'].values
pred_3 = pd.read_csv('stack_minmax_mean.csv')['is_churn'].values
pred_4 = pd.read_csv('stack_minmax_median.csv')['is_churn'].values
pred_5 = pd.read_csv('stack_minmax_bestbase.csv')['is_churn'].values
pred_6 = pd.read_csv('stack_minmax_minmax_medianbase.csv')['is_churn'].values

valid_true = pd.read_csv('train_pre_v3_201703.csv')['is_churn'].values

print('pred 1 log loss = {}'.format(log_loss(valid_true,pred_1)))
print('pred 2 log loss = {}'.format(log_loss(valid_true,pred_2)))
print('pred 3 log loss = {}'.format(log_loss(valid_true,pred_3)))
print('pred 4 log loss = {}'.format(log_loss(valid_true,pred_4)))
print('pred 5 log loss = {}'.format(log_loss(valid_true,pred_5)))
print('pred 6 log loss = {}'.format(log_loss(valid_true,pred_6)))