import pandas as pd
pd.read_csv("../input/key_2.csv",converters={'Page':lambda p:p[:-11]}, index_col='Page')\
        .join(pd.read_csv("../input/train_2.csv", usecols=[0]+list(range(669,729)), index_col='Page')\
        .median(axis=1,skipna=True).to_frame(name='Visits'), how='left')\
        .fillna(0).to_csv('oneline_submission_60.csv', float_format='%.0f', index=False)