# import  commod
import numpy as np
from PIL import Image
import itertools
from threading import Lock
import os
from cnn_code_tool import Code_tool
s='1234567890qwertyuiopasdfghjklzxcvbnm'
ct = Code_tool(s,
               'static/output/',
               'static/test/',
               'static/t/',
               lambda s: s.split('_')[1].split('.')[0],batch_size=64)
ct.train(30,0.5)
# ct.infer()
# print('ok')
# print(s(number+alphabet+ALPHABET))
