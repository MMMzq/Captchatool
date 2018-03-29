# import  commod
import numpy as np
from PIL import Image
import itertools
from threading import Lock
import os
from cnn_code_tool import Code_tool
# s='1234567890'
# s1='qwertyuiopasdfghjklzxcvbnm'
# s2='QWERTYUIOPASDFGHJKLZXCVBNM'
s='23456789qwertyupasdfghjkzxcvbnm'
ct = Code_tool(s,
               'static/output/',
               label_resolve_func=lambda s: s.split('_')[1].split('.')[0])
# ct.train(30,0.0001)
t=ct.infer('static/data/train/1_0_2.png')
print(t)

