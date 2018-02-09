import numpy as np
import pandas as pd
from datetime import datetime

members = pd.read_csv('../input/members_v3.csv')
registed_days = []
expired_days = []
total_days = []
birthdays = []
genders = []

base_datetime = datetime(2004, 3, 25)


def get_days(date):
    date = str(date)
    y = int(date[0:4])
    m = int(date[4:6])
    d = int(date[6:8])
    current_datetime = datetime(y, m, d)
    days = (current_datetime - base_datetime).days
    if(days < 1):
        days = -1
    return days

for i, row in members.iterrows():
    gender = row['gender']
    birthday = int(row['bd'])
    if(gender == None or gender == ''):
        gender = -1
    elif(gender == 'male'):
        gender = 1
    elif(gender == 'female'):
        gender = 2
    else:
        gender = 3
    if(birthday > 100 or birthday <8):
        birthday = -1
    birthdays.append(birthday)
    genders.append(gender)

pre_members = pd.DataFrame({
                'u_id': members['u_id'],
                'msno': members['msno'],
                'city': members['city'],
                'registered_via': members['registered_via']
})

pre_members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
pre_members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
pre_members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))
pre_members['birthdays'] = birthdays
pre_members['genders'] = genders

pre_members.to_csv('../input/pre_members.csv', index=False)


    
    

    