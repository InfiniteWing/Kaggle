from PIL import Image
bit=24
scale=4
offset=2
def color_trans(r, g, b):
    r=int(r/(256/bit))
    g=int(g/(256/bit))
    b=int(b/(256/bit))
    return r,g,b

fr=open("train.csv","r")
fw=open("pre/train_pre_origin_24.csv","w")
tags=["adult_males"]
for r in range(bit):
    for g in range(bit):
        for b in range(bit):
            tag="R{}G{}B{}".format(r,g,b)
            tags.append(tag)
outline=",".join(tags)
fw.writelines(outline+"\n")
fr.readline()
lines=fr.readlines()
for i,line in enumerate(lines):
    print(i)
    line=line.replace("\n","").split(",")
    id=line[0]
    adult_males=line[1]
    subadult_males=line[2]
    adult_females=line[3]
    juveniles=line[4]
    pups=line[5]
    im=Image.open("origin/Train/{}.jpg".format(id))
    width, height = im.size
    width=int(width/scale)
    height=int(height/scale)
    colors={}
    features=[adult_males]
    for r in range(bit):
        for g in range(bit):
            for b in range(bit):
                tag="R{}G{}B{}".format(r,g,b)
                colors[tag]=0
    for X in range(width):
        for Y in range(height):
            x=X*scale+offset
            y=Y*scale+offset
            r, g, b = im.getpixel((x, y))
            r, g, b = color_trans(r, g, b)
            tag="R{}G{}B{}".format(r,g,b)
            colors[tag]+=1
    for r in range(bit):
        for g in range(bit):
            for b in range(bit):
                tag="R{}G{}B{}".format(r,g,b)
                feature=str(colors[tag])
                features.append(feature)
    outline=",".join(features)
    fw.writelines(outline+"\n")