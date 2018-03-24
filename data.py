import requests
import commod
import os
from PIL import Image
import time
import numpy
import matplotlib.pyplot as plt

config=commod.read_config_yaml()
count=config['index']['original']
path=config['path']['original']
png_path=config['path']['png']
train_path=config['path']['data']['train']

def get_code_gif():
    url='http://www.fadeoooo.top/getGifCode.do'
    r=requests.get(url)
    commod.check_path_or_creat(path)
    f=open(path+str(count)+'.gif',mode='wb+')
    f.write(r.content)
    return f

def gif2png(f):
    commod.check_path_or_creat(png_path)
    with f:
        img=Image.open(f)
        img.seek(1)
        file_name=png_path+str(count)+'.png'
        img.save(file_name)
        return file_name

def png_resolve():
    file_list=commod.get_all_filename(png_path)
    x_lines = [10,44,83,114]
    y=0
    wight=32
    height=32
    train_index=1
    for fn in file_list:
        img=Image.open(png_path+fn)
        img=img.point(lambda i:i ==255,mode='1')
        for i in range(4):
            region=(x_lines[i],y,x_lines[i]+wight,y+height)
            im=img.crop(region)
            im.save(train_path + str(train_index)+'_'+str(i)+'_'+fn[i]+ '.png')
        train_index+=1


png_resolve()
