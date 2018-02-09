import numpy as np
import pandas as pd

trace_msno = 'hDW0/cSgIayezNOh5RsbKKskn4WsNUhrx6ZEbInTSQk='
transactions_v1 = pd.read_csv('../input/transactions.csv')
transactions_v2 = pd.read_csv('../input/transactions_v2.csv')
transactions = transactions_v1.append(transactions_v2).reset_index(drop=True)
transactions = transactions.sort_values(['msno','transaction_date'])
while True:
    trace_msno = input()
    print(transactions.loc[transactions['msno'] == trace_msno])
