from captcha.image import ImageCaptcha
from PIL import Image
import random
import os
from cnn_code_tool import Code_tool
def generate_todir(charset,charset_len,path,width,height,amount):
    if not os.path.exists(path):
        os.makedirs(path)
    ic=ImageCaptcha(width,height)
    for _ in range(amount):
        label=''.join(random.sample(charset,charset_len))
        img=ic.generate(label)
        with Image.open(img) as img:
            img.save(path+str(random.randint(1,amount))+'_'+label+'.png')

label_resolve=lambda filename:filename.split('_')[1].split('.')[0]

# 识别1个字符的demo
def demo1():
    charset='1234567890'
    train_path='data/demo1/train/'
    test_path='data/demo1/test/'
    model_path='data/demo1/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # 创建数据
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        generate_todir(charset,1,train_path,40,60,2000)
        generate_todir(charset,1,test_path,40,60,200)
    ct=Code_tool(charset,model_path,train_path,test_path,label_resolve)
    ct.train(20,0.8,1.,True)    # 迭代20代.或达到目标命中率退出训练,这里的命中率是针对单个字符的
    fn=os.listdir(test_path)[0]
    label=label_resolve(fn)
    code=ct.infer_file(test_path+fn)
    print('正确为:{}\t推测为:{}'.format(label,code))

# 识别1个字符,比demo1字符集更大
def demo2():
    charset='1234567890qwertyuiopasdfghjklzxcvbnm'
    train_path='data/demo2/train/'
    test_path='data/demo2/test/'
    model_path='data/demo2/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # 创建数据
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        generate_todir(charset,1,train_path,40,60,4000)
        generate_todir(charset,1,test_path,40,60,400)
    ct=Code_tool(charset,model_path,train_path,test_path,label_resolve)
    ct.train(20,0.8,0.75,True)  # 迭代20代.或达到目标命中率退出训练,这里的命中率是针对单个字符的
    fn=os.listdir(test_path)[0]
    label=label_resolve(fn)
    code=ct.infer_file(test_path+fn)
    print('正确为:{}\t推测为:{}'.format(label,code))

# 识别4个字符的demo
def demo3():
    charset='1234567890qwertyuiopasdfghjklzxcvbnm'
    train_path='data/demo3/train/'
    test_path='data/demo3/test/'
    model_path='data/demo3/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # 创建数据
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        generate_todir(charset,4,train_path,160,60,8000)
        generate_todir(charset,4,test_path,160,60,800)
    ct=Code_tool(charset,model_path,train_path,test_path,label_resolve)
    ct.train(30,0.75,0.8,True) # 迭代30代.或达到目标命中率退出训练,这里的命中率是针对单个字符的
    fn=os.listdir(test_path)[0]
    label=label_resolve(fn)
    code=ct.infer_file(test_path+fn)
    print('正确为:{}\t推测为:{}'.format(label,code))

# 从已有的模型继续训练的demo
def demo4():
    charset='1234567890'
    train_path='data/demo1/train/'
    test_path='data/demo1/test/'
    model_path='data/demo1/model/'
    ct=Code_tool(charset,model_path,train_path,test_path,label_resolve)
    ct.train(10,0.95,1.,False)

# 导入已有模型直接推测验证码,
def demo5():
    charset='1234567890'
    model_path='data/demo1/model/'
    test_path='data/demo1/test/'
    fn=os.listdir(test_path)[0]
    ct=Code_tool(charset,model_path)
    # 从bytes对象读取图片来推测
    with open(test_path+fn,'rb') as f:
        b=f.read() #只是为了展现infer_bytes()方法才这样做的,实际是不用的
        code=ct.infer_bytes(b)
        label=label_resolve(fn)
        print('正确为:{}\t推测为:{}'.format(label,code))


# 取消注释来运行其他demo
if __name__ == '__main__':
    demo1()
    # demo2()
    # demo3()
    # demo4()
    # demo5()
    pass