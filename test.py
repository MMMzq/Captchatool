# import  commod
import numpy as np
from PIL import Image
import itertools
from threading import Lock
import os
from cnn_code_tool import Code_tool
def s(shuzu):
    s=''
    for i in shuzu:
        s+=i
    return s
s='1234567890qwertyuiopasdfghjklzxcvbnm'
ct = Code_tool('e:/code/static/test/',
               'e:/code/static/t/',
               'e:/code/static/output/',
               s,
               lambda s: s.split('_')[1].split('.')[0],batch_size=64)
ct.train(20)
# print(s(number+alphabet+ALPHABET))
