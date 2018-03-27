#coding:utf-8
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random,time,os
import threading

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(captcha_size,char_set=number+alphabet+ALPHABET):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text

# 生成字符对应的验证码
def gen_captcha_text_and_image(i):
	image = ImageCaptcha()

	captcha_text = random_captcha_text(4)
	captcha_text = ''.join(captcha_text)

	captcha = image.generate(captcha_text)
	with Image.open(captcha) as img:
		captcha_image = img
		captcha_image=captcha_image.convert("L")
		captcha_image.save('static/f/test/'+str(i)+'_'+captcha_text+'.png')

def g(i):
	for _ in  range(250):
		gen_captcha_text_and_image(i)
		i+=1

if __name__ == '__main__':
	g(70000)
	# threading.Thread(target=g,args=(0),daemon=True)
	# threading.Thread(target=g,args=(10000),daemon=False)
	# threading.Thread(target=g,args=(20000),daemon=False)
	# threading.Thread(target=g,args=(30000),daemon=False)
	# threading.Thread(target=g,args=(40000),daemon=False)
	# threading.Thread(target=g,args=(50000),daemon=False)

