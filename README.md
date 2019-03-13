# Captcha-tool
用于识别普通验证码的工具类
## 安装
* 环境要求: python3
* **[点击我](https://github.com/MMMzq/Captchatool)**把项目克隆或直接下载
* 在项目目录下输入命令 `pip3 install -r r-cpu.txt`或者是`pip3 install -r r-gpu.txt`,后者会安装tensorflow-gpu版本.(强烈推荐gpu版本,gpu版本会在模型训练时会比cpu版本快很多,但是会多了一些安装步骤,具体请百度)

## 使用
* 导入模块
```python
    from cnn_code_tool import Code_tool
```
* 创建类
```python
    ct=Code_tool(charset,model_path,train_path,test_path,label_resolve)
    # charset 验证码的所有字符集
    # model_path 模型路径,训练完后模型将会保存在此路径下,推测验证码时模型将会从该路径下导入模型
    # train_path 训练数据路径,模型训练时使用
    # test_path 测试数据路径,模型训练时使用
    # label_resolve 从文件名解析出标签的函数,模型训练时使用
```
* 训练模型
```python
    ct.train()
```
* 训练完后推测验证码
```python
    captcha=ct.infer_file('captcha_img_filename') # 导入并推测验证码图片的验证码
```

## 帮助
* 上面的使用说明已经是一个完整的例子了,**更多运行例子可以运行项目demo,我已经添加了大量注释来说明**
* 一般完成一个验证码的识别的过程有如下步骤
    * 手工打码,把打码后的图片,放到训练路径下,测试图片同理(一般对于只含数字和26个字母长度为4的验证来说,并且可以切割出一个个字符的话,500~1000张打码后的图片即可达到0.8左右的命中率,如果不能分割的话一般需要几千张图片来训练模型)
    * 数据准备好后创建Code_tool类调用`train()`方法训练模型
    * 模型训练完后,一般调用`infer_file()`或者`infer_bytes()`来推测验证码,`infer_file()`用于从文件导入数据来推测其验证码,`infer_bytes()`一般用于网络获取验证码的字节对象后直接推测其验证码,例如使用requests库请求验证码后马上推测其验证码
    ```python
        r=requests.get('获取验证码的url')
        captcha=ct.infer_bytes(r.content)
        print(captcha) # 打印模型推测的验证码
    ```
* **值得注意的是推测的图片要和模型训练时的图片的宽度,高度,通道,格式要完全一致**
* 更多详尽说明[点击跳转](https://www.jianshu.com/p/354eb20942ea)
