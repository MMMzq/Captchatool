#coding:utf-8
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random,time,os

# 验证码中的字符, 就不用汉字了
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
i=17000
# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number+alphabet, captcha_size=1):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text

# 生成字符对应的验证码
def gen_captcha_text_and_image():
	image = ImageCaptcha()

	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)

	captcha = image.generate(captcha_text)
	with Image.open(captcha) as img:
		captcha_image = img
		captcha_image=captcha_image.convert("L").point(lambda b : b<240  , mode='1')
		captcha_image.save('h:/code/static/test/'+str(i)+'_'+captcha_text+'.png')


if __name__ == '__main__':
	for _ in  range(10000):
		gen_captcha_text_and_image()
		if i>22000:
			break
		i+=1

