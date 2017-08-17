import random
#read orders
fr = open("../input/orders.csv", 'r')
fr.readline()# skip header
lines=fr.readlines()
outcsv1 = open("../input/orders_valid_1_add.csv", 'w')
outcsv1.writelines("order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order"+"\n")
outcsv2 = open("../input/orders_valid_2_add.csv", 'w')
outcsv2.writelines("order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order"+"\n")
print(len(lines))
prior_count=0
for i,line in enumerate(lines):
    datas=line.replace("\n","").split(",")
    eval_set=(datas[2])
    
    try:
        datas1=lines[i+1].replace("\n","").split(",")
        eval_set1=(datas1[2])
    except:
        pass
    if(eval_set!='prior'):
        prior_count=0
    else:
        prior_count+=1
    if(eval_set1=='train'):
        if(random.randint(1,100)>50):
            outline=','.join(datas)
            outcsv2.writelines(outline+"\n")
            if(prior_count>4):
                datas[2]='train'
            outline=','.join(datas)
            outcsv1.writelines(outline+"\n")
        else:
            outline=','.join(datas)
            outcsv1.writelines(outline+"\n")
            if(prior_count>4):
                datas[2]='train'
            outline=','.join(datas)
            outcsv2.writelines(outline+"\n")
    else:
            
        if(eval_set=='train'):
            if(random.randint(1,100)>50):
                outline=','.join(datas)
                outcsv2.writelines(outline+"\n")
                datas[2]='valid'
                outline=','.join(datas)
                outcsv1.writelines(outline+"\n")
            else:
                outline=','.join(datas)
                outcsv1.writelines(outline+"\n")
                datas[2]='valid'
                outline=','.join(datas)
                outcsv2.writelines(outline+"\n")
                
        else:
            outline=','.join(datas)
            outcsv1.writelines(outline+"\n")
            outcsv2.writelines(outline+"\n")