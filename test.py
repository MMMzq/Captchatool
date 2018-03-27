# import  commod
import numpy as np
from PIL import Image
import itertools
from threading import Lock
import os
from cnn_code_tool import Code_tool
s='1234567890'
s1='qwertyuiopasdfghjklzxcvbnm'
s2='QWERTYUIOPASDFGHJKLZXCVBNM'
ct = Code_tool(s+s1+s2,
               'static/output/',
               'static/data/train/',
               'static/data/test/',
               lambda s: s.split('_')[2].split('.')[0],batch_size=64)
# ct.train(30,0.98,False)
ct.infer()
