from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics
orders={}
orders_pos={}
orders_dows={}
orders_hours={}
orders_new_product_rate={}
orders_none_order={}

products={}
users={}
users_products={}
products_aisles={}
products_departments={}

user_level={}
user_category={}
user_product_buytime_mean={}
user_product_buydow_mean={}
user_buytime_mean={}
user_buydow_mean={}
user_product_avg_cart_pos={}
user_distinct_department={}
user_distinct_aisle={}
user_new_product_rate={}
user_none_order_rate={}
user_product_highest_combo={}
user_product_current_combo={}
up_product_highest_combo_avg={}
up_product_buytime_mean={}
up_product_buydow_mean={}
ordered_in_past4_orders={}


print("Start loading products.csv")
products_df=pd.read_csv('../input/products.csv', encoding = 'UTF-8')
for index, row in tqdm(products_df.iterrows()):
    product_id=str(row['product_id'])
    aisles_id=str(row['aisle_id'])
    departments_id=(row['department_id'])
    products_aisles[product_id]=aisles_id
    products_departments[product_id]=departments_id
    #print(product_id,aisles_id,departments_id)


print("Start loading order_products__prior.csv")
#read prior orders
fr = open("../input/order_products__prior.csv", 'r', encoding = 'UTF-8')
fr.readline()# skip header
lines=fr.readlines()
for i,line in tqdm(enumerate(lines)):
    datas=line.replace("\n","").split(",")
    order_id=(datas[0])
    product_id=(datas[1])
    pos=int(datas[2])
    reorderer=int(datas[3])
    if(order_id not in orders):
        orders[order_id]={}
        orders_pos[order_id]={}
    orders[order_id][product_id]=reorderer
    orders_pos[order_id][product_id]=pos

print("Start loading order_products__train.csv")
#read train orders
fr = open("../input/order_products__train.csv", 'r', encoding = 'UTF-8')
fr.readline()# skip header
lines=fr.readlines()
for i,line in tqdm(enumerate(lines)):
    datas=line.replace("\n","").split(",")
    order_id=(datas[0])
    product_id=(datas[1])
    pos=int(datas[2])
    reorderer=int(datas[3])
    if(order_id not in orders):
        orders[order_id]={}
        orders_pos[order_id]={}
    orders[order_id][product_id]=reorderer
    orders_pos[order_id][product_id]=pos

for order_id in orders:
    reorder_list=list(orders[order_id].values())
    orders_new_product_rate[order_id]=(len(reorder_list)-sum(reorder_list))/len(reorder_list)
    if(sum(reorder_list)==0):
        orders_none_order[order_id]=1
    else:
        orders_none_order[order_id]=0
    
print("Start loading orders.csv")
#read orders
fr = open("../input/orders.csv", 'r', encoding = 'UTF-8')

fr.readline()# skip header
lines=fr.readlines()
for i,line in tqdm(enumerate(lines)):
    datas=line.replace("\n","").split(",")
    order_id=(datas[0])
    user_id=(datas[1])
    eval_set=(datas[2])
    order_number=(datas[3])
    order_dow=int(datas[4])
    order_hours=int(datas[5])
    orders_dows[order_id]=order_dow
    orders_hours[order_id]=order_hours
    if(user_id not in users):
        users[user_id]={}
        user_level[user_id]=0
    if(eval_set=="prior"):
        users[user_id][order_number]=order_id
        user_level[user_id]+=1
    elif(eval_set=="train"):
        users[user_id]["train"]=order_id
    if(eval_set=="test"):
        users[user_id]["test"]=order_id
    if(eval_set=="valid"):
        users[user_id]["valid"]=order_id


print("Start creating features")

for user_id in tqdm(users):
    users_products[user_id]=[]
    user_product_buytime_mean[user_id]={}
    user_product_buydow_mean[user_id]={}
    user_buytime_mean[user_id]=[]
    user_buydow_mean[user_id]=[]
    user_product_avg_cart_pos[user_id]={}
    user_distinct_department[user_id]={}
    user_distinct_aisle[user_id]={}
    user_product_highest_combo[user_id]={}
    user_product_current_combo[user_id]={}
    
    tmp_product_records={}
    tmp_last_4_products={}
    
    user_new_product_rate[user_id]=[]
    user_none_order_rate[user_id]=[]
    
    
    for i,(order_number, order_id) in enumerate(users[user_id].items()):
        if(order_number in ["test","train","valid"]):
            continue
        user_buytime_mean[user_id].append(orders_hours[order_id])
        user_buydow_mean[user_id].append(orders_dows[order_id])
        
        tmp_product_check={}
        
        for product_id in orders[order_id]:
            if(product_id not in users_products[user_id]):
                users_products[user_id].append(product_id)
            tmp_product_check[product_id]=1
            if(product_id not in user_product_buytime_mean[user_id]):
                user_product_buytime_mean[user_id][product_id]=[]
                user_product_buydow_mean[user_id][product_id]=[]
                user_product_avg_cart_pos[user_id][product_id]=[]
            user_product_buytime_mean[user_id][product_id].append(orders_hours[order_id])
            user_product_buydow_mean[user_id][product_id].append(orders_dows[order_id])
            user_product_avg_cart_pos[user_id][product_id].append(orders_pos[order_id][product_id])
            
            
            
            aisles_id=products_aisles[product_id]
            departments_id=products_departments[product_id]
            
            if(departments_id not in user_distinct_department[user_id]):
                user_distinct_department[user_id][departments_id]=1
            if(aisles_id not in user_distinct_aisle[user_id]):
                user_distinct_aisle[user_id][aisles_id]=1
            
            if(int(order_number)!=1):
                user_new_product_rate[user_id].append(orders_new_product_rate[order_id])
                user_none_order_rate[user_id].append(orders_none_order[order_id])
                
            if(product_id not in tmp_product_records):
                tmp_product_records[product_id]=[]
            tmp_product_records[product_id].append(1)
            
            if(len(users[user_id])-5>=i):
                if(product_id not in tmp_last_4_products):
                    tmp_last_4_products[product_id]=0
                tmp_last_4_products[product_id]+=1
        for product_id in tmp_product_records:
            if(product_id not in tmp_product_check):
                tmp_product_records[product_id].append(0)
    
    ordered_in_past4_orders[user_id]={}
    for product_id in tmp_last_4_products:
        ordered_in_past4_orders[user_id][product_id]=tmp_last_4_products[product_id]
    
    user_buytime_mean[user_id]=statistics.mean(user_buytime_mean[user_id])
    user_buydow_mean[user_id]=statistics.mean(user_buydow_mean[user_id])
    
    user_new_product_rate[user_id]=statistics.mean(user_new_product_rate[user_id])
    user_none_order_rate[user_id]=statistics.mean(user_none_order_rate[user_id])
    
    user_distinct_department[user_id]=len(user_distinct_department[user_id])
    user_distinct_aisle[user_id]=len(user_distinct_aisle[user_id])
    
    for product_id in user_product_buytime_mean[user_id]:
        user_product_buytime_mean[user_id][product_id]=statistics.mean(user_product_buytime_mean[user_id][product_id])
        user_product_buydow_mean[user_id][product_id]=statistics.mean(user_product_buydow_mean[user_id][product_id])
        user_product_avg_cart_pos[user_id][product_id]=statistics.mean(user_product_avg_cart_pos[user_id][product_id])
        
        if(product_id not in up_product_buytime_mean):
            up_product_buytime_mean[product_id]=[]
            up_product_buydow_mean[product_id]=[]
        up_product_buytime_mean[product_id].append(user_product_buytime_mean[user_id][product_id])
        up_product_buydow_mean[product_id].append(user_product_buydow_mean[user_id][product_id])
    
    # 計算combo
    for product_id,reorder_list in tmp_product_records.items():
        current_combo=0
        highest_combo=0
        for i in range(len(reorder_list)):
            if(i==0):
                if(reorder_list[i]==1):
                    current_combo=1
                continue
            if(reorder_list[i-1]==1 and reorder_list[i]==1):
                current_combo+=1
            elif(reorder_list[i-1]==1 and reorder_list[i]==0):
                if(current_combo>highest_combo):
                    highest_combo=current_combo
                current_combo=0
            elif(reorder_list[i]==1):
                current_combo=1
        if(current_combo>highest_combo):
            highest_combo=current_combo
        user_product_highest_combo[user_id][product_id]=highest_combo
        user_product_current_combo[user_id][product_id]=current_combo
        if(product_id not in up_product_highest_combo_avg):
            up_product_highest_combo_avg[product_id]=[]
        up_product_highest_combo_avg[product_id].append(highest_combo)

print("Start calculating overall features")
for product_id in up_product_highest_combo_avg:
    up_product_highest_combo_avg[product_id]=statistics.mean(up_product_highest_combo_avg[product_id])


for product_id in up_product_buytime_mean:    
    up_product_buytime_mean[product_id]=statistics.mean(up_product_buytime_mean[product_id])
    up_product_buydow_mean[product_id]=statistics.mean(up_product_buydow_mean[product_id])

outcsv = open("features/new_features.csv", 'w', encoding = 'UTF-8')


cols=[]
cols.append("user_id")
cols.append("product_id")

#cols.append("user_level")
cols.append("user_product_buytime_mean")
cols.append("user_product_buydow_mean")
cols.append("user_product_avg_cart_pos")

cols.append("user_buytime_mean")
cols.append("user_buydow_mean")
cols.append("user_distinct_department")
cols.append("user_distinct_aisle")

cols.append("user_new_product_rate")
cols.append("user_none_order_rate")
cols.append("user_product_highest_combo")
cols.append("user_product_current_combo")

cols.append("up_product_highest_combo_avg")
cols.append("up_product_buytime_mean")
cols.append("up_product_buydow_mean")
cols.append("ordered_in_past4_orders")

outcsv.writelines(','.join(cols)+"\n")
outcsv.flush()

print("Start saving features to csv")
for user_id in tqdm(users): 
    for product_id in users_products[user_id]:
        features=[]
        features.append(user_id)
        features.append(product_id)
        
        features.append(user_product_buytime_mean[user_id][product_id])
        features.append(user_product_buydow_mean[user_id][product_id])
        features.append(user_product_avg_cart_pos[user_id][product_id])
        
        
        features.append(user_buytime_mean[user_id])
        features.append(user_buydow_mean[user_id])
        features.append(user_distinct_department[user_id])
        features.append(user_distinct_aisle[user_id])
        
        features.append(user_new_product_rate[user_id])
        features.append(user_none_order_rate[user_id])
        features.append(user_product_highest_combo[user_id][product_id])
        features.append(user_product_current_combo[user_id][product_id])
        
        features.append(up_product_highest_combo_avg[product_id])
        features.append(up_product_buytime_mean[product_id])
        features.append(up_product_buydow_mean[product_id])
        if(product_id not in ordered_in_past4_orders[user_id]):
            features.append(0)
        else:
            features.append(ordered_in_past4_orders[user_id][product_id])
        
        
        features_str=[]
        for feature in features:
            if(isinstance(feature, float)):
                features_str.append(str(round(feature,5)))
            else:
                features_str.append(str(feature))
                
        
        outcsv.writelines(','.join(features_str)+"\n")
        outcsv.flush()
        