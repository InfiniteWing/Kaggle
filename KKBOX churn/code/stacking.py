import os
import numpy as np
import pandas as pd

sub_path = "subs"
all_files = os.listdir(sub_path)

# Read and concatenate submissions
outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
concat_sub = pd.concat(outs, axis=1)
cols = list(map(lambda x: "is_churn_" + str(x), range(len(concat_sub.columns))))
concat_sub.columns = cols
concat_sub.reset_index(inplace=True)
print(concat_sub.head())
# check correlation
print(concat_sub.corr())
concat_sub['is_churn_mean'] = concat_sub.iloc[:, 1:4].mean(axis=1)
concat_sub['is_churn'] = concat_sub['is_churn_mean']
concat_sub[['msno', 'is_churn']].to_csv('stack_mean.csv', 
                                        index=False)
'''
# get the data fields ready for stacking
concat_sub['is_churn_max'] = concat_sub.iloc[:, 1:4].max(axis=1)
concat_sub['is_churn_min'] = concat_sub.iloc[:, 1:4].min(axis=1)
concat_sub['is_churn_mean'] = concat_sub.iloc[:, 1:4].mean(axis=1)
concat_sub['is_churn_median'] = concat_sub.iloc[:, 1:4].median(axis=1)
# set up cutoff threshold for lower and upper bounds, easy to twist 
cutoff_lo = 0.8
cutoff_hi = 0.2

concat_sub['is_churn'] = concat_sub['is_churn_mean']
concat_sub[['msno', 'is_churn']].to_csv('stack_mean.csv', 
                                        index=False, float_format='%.6f')
                                        
concat_sub['is_churn'] = concat_sub['is_churn_median']
concat_sub[['msno', 'is_churn']].to_csv('stack_median.csv', 
                                        index=False, float_format='%.6f')
                                        
                                        
concat_sub['is_churn'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    concat_sub['is_churn_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_churn_min'], 
                                             concat_sub['is_churn_mean']))
concat_sub[['msno', 'is_churn']].to_csv('stack_minmax_mean.csv', 
                                        index=False, float_format='%.6f')

concat_sub['is_churn'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    concat_sub['is_churn_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_churn_min'], 
                                             concat_sub['is_churn_median']))
concat_sub[['msno', 'is_churn']].to_csv('stack_minmax_median.csv', 
                                        index=False, float_format='%.6f')

# load the model with best base performance
sub_base = pd.read_csv('subs/lgb_sub.csv')

concat_sub['is_churn_base'] = sub_base['is_churn']
concat_sub['is_churn'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    concat_sub['is_churn_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_churn_min'], 
                                             concat_sub['is_churn_base']))
concat_sub[['msno', 'is_churn']].to_csv('stack_minmax_bestbase.csv', 
                                        index=False, float_format='%.6f')

# load the model with best minmax_median base performance
sub_base = pd.read_csv('stack_minmax_median.csv')

concat_sub['is_churn_base'] = sub_base['is_churn']
concat_sub['is_churn'] = np.where(np.all(concat_sub.iloc[:,1:4] > cutoff_lo, axis=1), 
                                    concat_sub['is_churn_max'], 
                                    np.where(np.all(concat_sub.iloc[:,1:4] < cutoff_hi, axis=1),
                                             concat_sub['is_churn_min'], 
                                             concat_sub['is_churn_base']))
concat_sub[['msno', 'is_churn']].to_csv('stack_minmax_minmax_medianbase.csv', 
                                        index=False, float_format='%.6f')        
'''                                        