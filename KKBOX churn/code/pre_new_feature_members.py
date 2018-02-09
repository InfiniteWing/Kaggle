import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
members = pd.read_csv('../input/pre_members.csv')

members_list = list(members['msno'])

members_records = {}
for msno in members_list:
    members_records[msno] = {}
    members_records[msno]['train'] = 0
    members_records[msno]['test'] = 0
    members_records[msno]['total_play'] = 0
    members_records[msno]['total_replay'] = 0
    members_records[msno]['replay_rate'] = 0

for i, row in tqdm(train.iterrows(), total = len(train)):
    msno = row['msno']
    target = int(row['target'])
    members_records[msno]['train'] += 1
    members_records[msno]['total_play'] += 1
    
    if(target == 1):
        members_records[msno]['total_replay'] += 1
        
for i, row in tqdm(test.iterrows(), total = len(test)):
    msno = row['msno']
    members_records[msno]['test'] += 1
        
            
print("Creating members features..")
total_plays = []
total_replays = []
replay_rates = []
train_counts = []
test_counts = []

for msno in members_list:
    if(members_records[msno]['total_play'] == 0):
        members_records[msno]['total_play'] = -1
        members_records[msno]['total_replay'] = -1
        members_records[msno]['replay_rate'] = -1
    else:
        
        members_records[msno]['replay_rate'] = members_records[msno]['total_replay'] / members_records[msno]['total_play']
    total_plays.append(members_records[msno]['total_play'])
    total_replays.append(members_records[msno]['total_replay'])
    replay_rates.append(members_records[msno]['replay_rate'])
    train_counts.append(members_records[msno]['train'])
    test_counts.append(members_records[msno]['test'])
    
members['member_total_play'] = total_plays
members['member_total_replay'] = total_replays
members['member_replay_rate'] = replay_rates
members['train'] = train_counts
members['test'] = test_counts
members.to_csv('../input/pre_members_2.csv', index=False)

