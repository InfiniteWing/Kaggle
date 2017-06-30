import math

fr = open("train.csv", 'r')
fw = open("train_fix.csv", 'w')
index=0
header=fr.readline().replace("\n","")
lines=fr.readlines()
fw.writelines(header+"\n")
for line in lines:
	data=line.replace("\n","").split(',')
	for i,d in enumerate(data):
		if(d=="" or d=="NA"):
			data[i]="-1"
	fw.writelines(','.join(data)+"\n")
fr.close()
fw.close()
fr = open("test.csv", 'r')
fw = open("test_fix.csv", 'w')
index=0
header=fr.readline().replace("\n","")
lines=fr.readlines()
fw.writelines(header+"\n")
for line in lines:
	data=line.replace("\n","").split(',')
	for i,d in enumerate(data):
		if(d=="" or d=="NA"):
			data[i]="-1"
	fw.writelines(','.join(data)+"\n")
fr.close()
fw.close()
