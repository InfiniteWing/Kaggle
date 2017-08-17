from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
orders={}
products={}
users={}
users_products={}
products_aisles={}
products_departments={}

departments_total_buys={}
departments_total_first_time_buys={}
departments_total_second_time_buys={}
departments_overall_reorder_rate={}

aisles_total_buys={}
aisles_total_first_time_buys={}
aisles_total_second_time_buys={}
aisles_overall_reorder_rate={}

user_departments_total_first_time_buys={}
user_departments_total_second_time_buys={}
user_departments_total_buys={}
user_departments_reorder_rate={}

user_aisles_total_first_time_buys={}
user_aisles_total_second_time_buys={}
user_aisles_total_buys={}
user_aisles_reorder_rate={}

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
	reorderer=int(datas[3])
	if(order_id not in orders):
		orders[order_id]={}
	orders[order_id][product_id]=reorderer


print("Start loading order_products__train.csv")
#read train orders
fr = open("../input/order_products__train.csv", 'r', encoding = 'UTF-8')
fr.readline()# skip header
lines=fr.readlines()
for i,line in tqdm(enumerate(lines)):
	datas=line.replace("\n","").split(",")
	order_id=(datas[0])
	product_id=(datas[1])
	reorderer=int(datas[3])
	if(order_id not in orders):
		orders[order_id]={}
	orders[order_id][product_id]=reorderer
  
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
	if(user_id not in users):
		users[user_id]={}
	if(eval_set=="prior"):
		users[user_id][order_number]=order_id
	elif(eval_set=="train"):
		users[user_id]["train"]=order_id
	if(eval_set=="test"):
		users[user_id]["test"]=order_id
	if(eval_set=="valid"):
		users[user_id]["valid"]=order_id


print("Start creating features")
user_distinct_products={}
for user_id in tqdm(users):
	user_departments_total_first_time_buys[user_id]={}
	user_departments_total_second_time_buys[user_id]={}
	user_departments_total_buys[user_id]={}
	user_departments_reorder_rate[user_id]={}
	
	user_aisles_total_first_time_buys[user_id]={}
	user_aisles_total_second_time_buys[user_id]={}
	user_aisles_total_buys[user_id]={}
	user_aisles_reorder_rate[user_id]={}
	
	products_check_stack={}
	
	for order_number, order_id in users[user_id].items():
		if(order_number in ["test","train","valid"]):
			continue
		for product_id in orders[order_id]:
		
			aisles_id=products_aisles[product_id]
			departments_id=products_departments[product_id]
			
			if(departments_id not in departments_total_buys):
				departments_total_buys[departments_id]=0
				departments_total_first_time_buys[departments_id]=0
				departments_total_second_time_buys[departments_id]=0
				
			if(aisles_id not in aisles_total_buys):
				aisles_total_buys[aisles_id]=0
				aisles_total_first_time_buys[aisles_id]=0
				aisles_total_second_time_buys[aisles_id]=0
			
				
			if(departments_id not in user_departments_total_first_time_buys[user_id]):
				user_departments_total_first_time_buys[user_id][departments_id]=0
				user_departments_total_second_time_buys[user_id][departments_id]=0
				user_departments_total_buys[user_id][departments_id]=0
				user_departments_reorder_rate[user_id][departments_id]=0
						
			if(aisles_id not in user_aisles_total_first_time_buys[user_id]):
				user_aisles_total_first_time_buys[user_id][aisles_id]=0
				user_aisles_total_second_time_buys[user_id][aisles_id]=0
				user_aisles_total_buys[user_id][aisles_id]=0
				user_aisles_reorder_rate[user_id][aisles_id]=0
				
				
			aisles_total_buys[aisles_id]+=1
			departments_total_buys[departments_id]+=1
			user_aisles_total_buys[user_id][aisles_id]+=1
			user_departments_total_buys[user_id][departments_id]+=1
				
			if(product_id not in products_check_stack):
				# 第一次購買
				products_check_stack[product_id]=1
				
				departments_total_first_time_buys[departments_id]+=1
				aisles_total_first_time_buys[aisles_id]+=1
				user_aisles_total_first_time_buys[user_id][aisles_id]+=1
				user_departments_total_first_time_buys[user_id][departments_id]+=1
				
				
				
			elif(products_check_stack[product_id]==1):
				# 第二次購買
				products_check_stack[product_id]=2
				
				departments_total_second_time_buys[departments_id]+=1
				aisles_total_second_time_buys[aisles_id]+=1
				user_aisles_total_second_time_buys[user_id][aisles_id]+=1
				user_departments_total_second_time_buys[user_id][departments_id]+=1
	
	user_distinct_products[user_id]=products_check_stack
departments_check_stack={}
aisles_check_stack={}

outcsv = open("departments_aisles_features.csv", 'w', encoding = 'UTF-8')
cols=[]
cols.append("user_id")
cols.append("product_id")
cols.append("departments_id")
cols.append("aisles_id")

cols.append("user_departments_total_first_time_buys")
cols.append("user_departments_total_second_time_buys")
cols.append("user_departments_total_buys")
cols.append("user_departments_reorder_rate")

cols.append("user_aisles_total_first_time_buys")
cols.append("user_aisles_total_second_time_buys")
cols.append("user_aisles_total_buys")
cols.append("user_aisles_reorder_rate")

cols.append("departments_total_buys")
cols.append("departments_total_first_time_buys")
cols.append("departments_total_second_time_buys")
cols.append("departments_overall_reorder_rate")

cols.append("aisles_total_buys")
cols.append("aisles_total_first_time_buys")
cols.append("aisles_total_second_time_buys")
cols.append("aisles_overall_reorder_rate")

outcsv.writelines(','.join(cols)+"\n")
outcsv.flush()

print("Start saving features to csv")
for user_id in tqdm(users):	
	products_check_stack=user_distinct_products[user_id]
	for product_id in products_check_stack:
		aisles_id=products_aisles[product_id]
		departments_id=products_departments[product_id]
		
		user_aisles_reorder_rate[user_id][aisles_id]=(
					(user_aisles_total_buys[user_id][aisles_id]-
					user_aisles_total_first_time_buys[user_id][aisles_id])
					/user_aisles_total_buys[user_id][aisles_id])
					
		
		user_departments_reorder_rate[user_id][departments_id]=(
					(user_departments_total_buys[user_id][departments_id]-
					user_departments_total_first_time_buys[user_id][departments_id])
					/user_departments_total_buys[user_id][departments_id])
					
		if(departments_id not in departments_check_stack):
			departments_check_stack[departments_id]=True
			departments_overall_reorder_rate[departments_id]=(
						(departments_total_buys[departments_id]-
						departments_total_first_time_buys[departments_id])
						/departments_total_buys[departments_id])
		if(aisles_id not in aisles_check_stack):	
			aisles_check_stack[aisles_id]=True
			aisles_overall_reorder_rate[aisles_id]=(
						(aisles_total_buys[aisles_id]-
						aisles_total_first_time_buys[aisles_id])
						/aisles_total_buys[aisles_id])
		
		
		features=[]
		features.append(user_id)
		features.append(product_id)
		features.append(departments_id)
		features.append(aisles_id)
		
		
		features.append(user_departments_total_first_time_buys[user_id][departments_id])
		features.append(user_departments_total_second_time_buys[user_id][departments_id])
		features.append(user_departments_total_buys[user_id][departments_id])
		features.append(user_departments_reorder_rate[user_id][departments_id])

		features.append(user_aisles_total_first_time_buys[user_id][aisles_id])
		features.append(user_aisles_total_second_time_buys[user_id][aisles_id])
		features.append(user_aisles_total_buys[user_id][aisles_id])
		features.append(user_aisles_reorder_rate[user_id][aisles_id])
		
		features.append(departments_total_buys[departments_id])
		features.append(departments_total_first_time_buys[departments_id])
		features.append(departments_total_second_time_buys[departments_id])
		features.append(departments_overall_reorder_rate[departments_id])
		
		
		features.append(aisles_total_buys[aisles_id])
		features.append(aisles_total_first_time_buys[aisles_id])
		features.append(aisles_total_second_time_buys[aisles_id])
		features.append(aisles_overall_reorder_rate[aisles_id])
		
		features_str=[]
		for feature in features:
			if(isinstance(feature, float)):
				features_str.append(str(round(feature,5)))
			else:
				features_str.append(str(feature))
				
		
		outcsv.writelines(','.join(features_str)+"\n")
		outcsv.flush()
		
		