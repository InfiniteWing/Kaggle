from PIL import Image
import os
bit=24
def color_trans(r, g, b):
    r=int(r/(256/bit))
    g=int(g/(256/bit))
    b=int(b/(256/bit))
    return r,g,b
scale=4
offset=2
test_index=5
test_size=3110
fw=open("pre/test_pre_origin_24_{}.csv".format(test_index),"w")
tags=["test_id"]
for r in range(bit):
    for g in range(bit):
        for b in range(bit):
            tag="R{}G{}B{}".format(r,g,b)
            tags.append(tag)
outline=",".join(tags)
fw.writelines(outline+"\n")
for index in range(test_size):
    i=index+test_index*test_size
    id=str(i)
    print(id)
    path="origin/Test/{}.jpg".format(id)
    if(not os.path.isfile(path)):
        break
    im=Image.open(path)
    width, height = im.size
    width=int(width/scale)
    height=int(height/scale)
    colors={}
    features=[id]
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
    fw.flush()