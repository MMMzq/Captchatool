import os
import random
from queue import Queue
from PIL import Image
import numpy as np
import traceback
import tensorflow as tf
import threading
from io import BytesIO,StringIO

'''
1.通常loss，和命中率是负相关的，loss降低一般意味着的命中率的提升！
2.在相同的模型中条件下，模型的大小，实际很大程度取决于图片的大小，和输入到全连接处时shape的大小，
    例如输入到全连接层shape为(batch_size,15,40,1)和(batch_size,8,20,1),后者的模型可能将会比前者小3/2
    其实，输入的图片宽度和高度越大，到全连接层的shape就会越大，
    所以想要模型变小，就要尽量减小输入到全连接层shape的大小，或者取消全连接层。
    又或者是增加层数来下采样减小到输入到全连接层shape的大小
3.请确保训练集和测试集数据的宽度,高度,通道数,格式保持一致
4.强烈建议安装tensorflow-gpu版本
5.如果验证长度不定,可以用不在验证码字符集一个的字符来补全到最大验证码长度
6.如果有什么问题可以在 https://github.com/MMMzq/Captchatool 提出issues
'''
lock = threading.Lock()
class Code_tool:
    # 标签解析函数
    __label_resolve_func = None
    # 训练集的路径
    __input_path = None
    # 测试集的路径
    __test_input_path = None
    # 模型路径
    __model_path = None
    # 验证码字符集
    __charset = None
    # 一个批次大小
    __batch_size = None
    # 文件名队列
    __filename_queue = None
    # 文件队列
    __file_queue = None
    # 数据队列
    __data_queue = None
    # 测试队列
    __test_queue = None
    # 最大验证码长度
    __max_captcha_len = None
    # 验证码字符集长度
    __charset_len = None
    # 训练集的路径下文件名列表
    __fnl = None
    # 测试集的路径下文件名列表
    __test_fnl=None
    # 用于是否持久session
    __isinit=False
    # 图片宽度
    __width = None
    # 图片高度
    __height = None
    # 图片通道数
    __channel = None
    # ------------------------------------------------------
    # 利用列表的append,remove的原子操作来在多线程环境下计数
    __filename_queue_handle_count = []
    __file_queue_handle_count = []
    __data_queue_handle_count = []

    '''
    Note:   如果只想从已有模型导入且只想用模型推断验证码，只需填charset,model_path参数
    参数:    
        charset: 验证码字符集
        model_path: 模型路径,train方法把训练好的模型保持此路径下,infer系列方法,将会从该路径下读取模型
        input_path: 训练集的路径
        test_input_path: 测试集的路径
        label_resolve_func: 从文件名解析出标签的函数,如果为None,则默认文件名就是标签
        batch_size: 生产一批次训练批次时所需的图片数
        test_batch_size: 生产一批次测试批次时所需的图片数,
                         如果测试文件夹目录文件不足生成一个测试批次，则用test目录下所有文件来测试
        max_thread_size: 最大线程数
        batch_multiple:  队列大小倍数,队列大小等于batch_size*batch_multiple
        data_handle_n:   组装数据队列处理者个数
    '''

    def __init__(self, charset, model_path, input_path=None, test_input_path=None,
                 label_resolve_func=None, batch_size=64, test_batch_size=128,
                 max_thread_size=16, batch_multiple=4, data_handle_n=3):
        self.__model_path = model_path
        self.__charset = charset
        self.__charset_len = len(charset)
        # 判断是否有路径，如果没有，将不能使用train方法！
        if input_path and test_input_path :
            self.__input_path = input_path
            self.__test_input_path = test_input_path
            self.__label_resolve_func = label_resolve_func or (lambda s: s)
            self.__batch_size = batch_size
            self.__data_handle_n = data_handle_n
            self.__max_thread_size = max_thread_size
            self.__filename_queue = Queue(self.__batch_size * batch_multiple)
            self.__file_queue = Queue(self.__batch_size * batch_multiple)
            self.__data_queue = Queue(self.__batch_size)
            self.__test_queue=Queue(32)
            self.__fnl = os.listdir(self.__input_path)
            self.__test_fnl=os.listdir(self.__test_input_path)
            self.__test_batch_size=min(test_batch_size,len(self.__test_fnl))
            # 获取图片形状
            fn=self.__input_path + self.__fnl[0]
            with Image.open(fn) as img:
                shape = np.array(img).shape
            self.__height = shape[0]
            self.__width = shape[1]
            if len(shape) == 2:
                self.__channel = 1
            else:
                self.__channel = shape[2]
            # 获取验证码长度
            self.__max_captcha_len = len(self.__label_resolve_func(self.__fnl[0]))

    '''
    Note:   将有一个文件名队列处理线程,一个测试队列,data_handle_n个组装数据队列处理者线程，其余剩下的线程都是文件队列处理者
            注意对应的处理者请不要超过对应队列的大小
    参数:   
        epochs: 等同于train方法的epochs参数,表示迭代次数 
    '''
    def __startwork(self, epochs):
        lock.acquire()
        self.__filename_queue_handle_count.append(None)
        t = threading.Thread(target=self.__filename_queue_handle, daemon=True,
                             args=(epochs,))
        t.start()
        t = threading.Thread(target=self.__test_quque_handle, daemon=True)
        t.start()
        for _ in range(self.__data_handle_n):
            self.__data_queue_handle_count.append(None)
            t = threading.Thread(target=self.__data_queue_handle, daemon=True)
            t.start()
        for _ in range(self.__max_thread_size - self.__data_handle_n - 2):
            self.__file_queue_handle_count.append(None)
            t = threading.Thread(target=self.__file_queue_handle, daemon=True)
            t.start()
        lock.release()
    '''
    Note:   清理工作
    '''
    def __endwork(self):
        self.__file_queue.queue.clear()
        self.__filename_queue.queue.clear()
        self.__data_queue.queue.clear()
        self.__data_queue_handle_count.clear()
        self.__file_queue_handle_count.clear()
        self.__filename_queue_handle_count.clear()

    '''
    ---------------------------训练数据读取处理-------------------------------
    '''
    '''
    Note:   将self.__fnl的文件名列表不断put进filename_queue队列,
            generate_next_batch方法生产的batch是乱序的,
            原因是有多个线程竞争从filename_queue获取文件名
    参数:
        epochs: 等同于train方法的epochs参数,表示迭代次数
    '''
    def __filename_queue_handle(self, epochs):
        try:
            for _ in range(epochs):
                for i in self.__fnl:
                    self.__filename_queue.put(i)
        except BaseException:
            msg = traceback.format_exc()
            print(msg)
            raise BaseException
        finally:
            self.__filename_queue_handle_count.remove(None)
            if len(self.__filename_queue_handle_count) == 0:
                self.__filename_queue.put(None) # 全部读取完后，向队列加入None用来告诉消费者已经到末尾了

    '''
    Note:   从filename_queue获取图片文件名,读取图片文件,
            并从文件名解析出标签,之后把标签文本转为向量,最后把一个图片numpy,标签numpy对象put进file_queue队列
    raise:  如果函数因抛出异常而提前退出函数，将会导致数据读取不完全！
    '''
    def __file_queue_handle(self):
        try:
            while True:
                file_name = self.__filename_queue.get()
                if file_name is None:
                    break   # 如果进入该块，表示已经没有数据可读了
                with Image.open(self.__input_path + file_name) as f:
                    img_array = np.array(f)
                    if img_array.shape[1] != self.__width or img_array.shape[0] != self.__height:
                        continue    # 格式不符合弃掉
                    img_raw = img_array.flatten()
                    fn = self.__label_resolve_func(file_name)
                    label_raw = self.__text2vec(fn)
                    self.__file_queue.put({'img_raw': img_raw, 'label_raw': label_raw}, )
        except BaseException:
            msg = traceback.format_exc()
            print(msg)
            raise BaseException
        finally:
            self.__file_queue_handle_count.remove(None)
            if len(self.__file_queue_handle_count) == 0:
                self.__file_queue.put(None)  # 发出数据处理者线程该结束的信号
            else:
                self.__filename_queue.put(None)  # 通知别的文件处理者线程结束的线程结束

    '''
    Note:   从file_queue队列获取多个图片numpy对象进行拼装成一个批次,最后put进data_queue队列    
    raise:  如果函数因抛出异常而提前退出函数，将会导致数据读取不完全！
    '''
    def __data_queue_handle(self):
        try:
            while True:
                batch_img = np.array([])
                batch_label = np.array([])
                for _ in range(self.__batch_size):
                    data = self.__file_queue.get()
                    if data is None:
                        if batch_label.size != 0:
                            self.__do_add_data_queue(batch_img, batch_label)
                        return
                    batch_img = np.append(batch_img, data['img_raw'])
                    batch_label = np.append(batch_label, data['label_raw'])
                self.__do_add_data_queue(batch_img, batch_label)
        except BaseException:
            msg = traceback.format_exc()
            print(msg)
            raise BaseException
        finally:
            self.__data_queue_handle_count.remove(None)
            if len(self.__data_queue_handle_count) == 0:
                # 向data_queue队列put进Node,之后如果在调用generate_next_batch方法将会抛出NullDataException
                self.__data_queue.put(None)
            else:
                self.__file_queue.put(None) #通知别的data_queue_handle结束

    '''
    Note:   通用方法
    '''
    def __do_add_data_queue(self, batch_img, batch_label):
        batch_label = np.reshape(batch_label, [-1, self.__max_captcha_len * self.__charset_len])
        batch_img = np.reshape(batch_img, [-1, self.__height * self.__width * self.__channel])
        self.__data_queue.put({'batch_img': batch_img, 'batch_label': batch_label}, )

    '''
    Note:   不断向test_queue队列put数据,一般频率不高
    '''
    def __test_quque_handle(self):
        while(True):
            self.__test_queue.put(self.__generate_test_batch())

    '''
    Note:   该方法不一定会返回等同与batch_size参数的batch,例如最后一批可能只有10张图片组成    
    return: 返回一批训练集
    raise:  当完全没有数据可读取完全是抛出,既从队列收到Node对象
    '''
    def __generate_next_batch(self):
        data = self.__data_queue.get()
        if data == None:
            raise NullDataException('空数据，原因所有数据已被读取')
        return data['batch_img'], data['batch_label']

    '''
    Note:   test文件夹和train文件夹一定不要相同！！！否则会影响测试命中率
    return: 返回一批测试集
    '''
    def __generate_test_batch(self):
        batch_img = np.array([])
        batch_label = np.array([])
        for fn in random.sample(self.__test_fnl,self.__test_batch_size):
            with Image.open(self.__test_input_path + fn) as f:
                img_array = np.array(f)
                if img_array.shape[1] != self.__width or img_array.shape[0] != self.__height:
                    continue    # 格式不符合弃掉
                img_raw = img_array.flatten()
                fn = self.__label_resolve_func(fn)
                label_raw = self.__text2vec(fn)
                batch_img = np.append(batch_img, img_raw)
                batch_label = np.append(batch_label, label_raw)
        batch_label = np.reshape(batch_label, [-1, self.__max_captcha_len * self.__charset_len])
        batch_img = np.reshape(batch_img, [-1, self.__height * self.__width * self.__channel])
        return batch_img, batch_label

    '''
    Note:   文本转向量
    参数:
        text:    一个str对象,请确保text的字符在self.__charset里有
    return: 返回一个numpy对象
    '''
    def __text2vec(self, text):
        vec = np.zeros([self.__max_captcha_len, self.__charset_len])
        for i in range(self.__max_captcha_len):
            index = self.__charset.find(text[i])
            vec[i, index] = 1
        vec = vec.flatten()
        return vec
    '''
    Note:   向量转文本
    参数:
        vec:    一个numpy对象,其中vec.shape要等于[captcha_len,charset_len],否则会错误
    return: 返回文本,即验证码
    '''
    def __vec2text(self, vec):
        # vec.shape 要等于[captcha_len,charset_len]，否则会不正确
        indexs = vec.argmax(axis=1)
        text = ''
        for i in indexs:
            text += self.__charset[i]
        return text

    '''
    ---------------------------cnn定义---------------------------------------
    '''
    '''
    Note:   定义模型,该模型有三层卷积层,一层全连接层,一层输出层,
            由于没有返回值,所以只能通过定义时的name属性获取或调用,怎么调用请看train方法的实际代码
    '''
    def __init_cnn(self):
        #这里的常量用于以后推测验证码时使用
        tf.constant(self.__height, name='height')
        tf.constant(self.__width, name='width')
        tf.constant(self.__channel, name='channel')
        tf.constant(self.__max_captcha_len,name='captcha_len')
        tf.constant(self.__charset_len,name='charset_len')

        # 定义占位符，变量,常量
        img_b = tf.placeholder(tf.float32, [None, self.__height * self.__width * self.__channel], name='img_p')
        label_b = tf.placeholder(tf.float32, name='label_p')
        keed = tf.placeholder(tf.float32, name='k_p')
        global_step = tf.Variable(0, name='g_v')
        img_b = tf.reshape(img_b, shape=[-1, self.__height, self.__width, self.__channel])
        w_alpha =0.01
        b_alpha =0.1

        # 定义模型
        def conv2(input, ksize,w_name=None,b_name=None,return_name=None, padding='SAME'):
            w = tf.Variable(w_alpha * tf.random_normal(ksize),name=w_name)
            b = tf.Variable(b_alpha * tf.random_normal([ksize[3]]),name=b_name)
            return tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding=padding), b,name=return_name)
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 卷积层,下面同理
        h = conv2(img_b, [3, 3, self.__channel, 32],'weight','bais')
        h = tf.nn.relu(h)
        h = max_pool_2x2(h)
        h = tf.nn.dropout(h, keed)

        h = conv2(h, [3, 3, 32, 64],'weight','bais')
        h = tf.nn.relu(h)
        h = max_pool_2x2(h)
        h = tf.nn.dropout(h, keed)

        h = conv2(h, [3, 3, 64, 64],'weight','bais')
        h = tf.nn.relu(h)
        h = max_pool_2x2(h)
        h = tf.nn.dropout(h, keed)

        # 全连接层
        shape = h.get_shape().as_list()
        dense=conv2(h,[shape[1] , shape[2] , shape[3],1024],'fc_weight','fc_bais',padding='VALID')
        dense = tf.nn.relu(dense)
        dense = tf.nn.dropout(dense, keed)

        # 输出层
        out=conv2(dense,[1,1,1024,self.__max_captcha_len * self.__charset_len],'out_weight','out_bais',return_name='out')
        out=tf.reshape(out,[-1,self.__max_captcha_len * self.__charset_len])

        #   定义操作
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_b, logits=out),
                              name='loss')
        train_op = tf.train.AdamOptimizer().minimize(loss, name='train')
        predict = tf.reshape(out, [-1, self.__max_captcha_len, self.__charset_len])
        label_b = tf.reshape(label_b, [-1, self.__max_captcha_len, self.__charset_len])
        max_logits_indexs = tf.argmax(predict, 2)
        max_label_indexs = tf.argmax(label_b, 2)
        correct_pred = tf.equal(max_label_indexs, max_logits_indexs)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


    '''
    Note: 该方法请不要在多线程环境运行,请确保同一时间内只有一个线程运行该方法
    参数:
        epochs:训练集迭代次数
        target_ac:  target_ac目标命中率,当模型达到该命中率,且不管是否达到迭代次数都会退出训练
        retrain:    True表示重新训练,False表示导入self.__model_path路径下最近更新的一个模型继续训练
        keep_prob:  即tf.nn.dropout(x, keep_prob)的第二个参数,该参数在测试集测试时将不会生效
    '''
    def train(self, epochs=10, target_ac=1.,keep_prob=1.,retrain=True):
        if self.__input_path is None or self.__test_input_path is None:
            print('__input_path或__test_input_path缺失')
            return
        self.__startwork(epochs)
        #   开始训练
        with tf.Session() as sess:
            if retrain:
                self.__init_cnn()
                save = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                step = 0
            else:
                cp = tf.train.latest_checkpoint(self.__model_path)
                save = tf.train.import_meta_graph(cp + '.meta')
                save.restore(sess, cp)
                step = int(sess.run('g_v:0'))
            while True:
                try:
                    img, label = self.__generate_next_batch()
                except NullDataException:
                    break
                loss, _ = sess.run(['loss:0', 'train'],
                                   feed_dict={'img_p:0': img, 'label_p:0': label, 'k_p:0': keep_prob})
                print('步数为：{}\tloss:{}'.format(step, loss))
                if step % 50 == 0:
                    test_img, test_label = self.__test_queue.get()
                    # 这里的命中率是针对单个字符的不代表真正的命中率
                    actual_ac = sess.run('accuracy:0',
                                         feed_dict={'img_p:0': test_img, 'label_p:0': test_label,
                                                    'k_p:0': 1.})
                    print('步数为：{}--------------------------------------命中率:{}'.format(step, actual_ac))
                    if actual_ac >= target_ac:
                        break
                step += 1
            # 保存
            tf.summary.FileWriter(self.__model_path, sess.graph)
            g_step = tf.get_default_graph().get_tensor_by_name('g_v:0')
            tf.assign(g_step, step, name='update')
            sess.run('update:0')
            self.__endwork()
            print('保存模型中请等待!')
            save.save(sess, self.__model_path + 'model', global_step=step)
            print('完成')

    '''
    Note:   持久化tensorlfow.Session对象,防止用模型推测时反复打开关闭session,导致浪费资源
    '''
    def __init_session(self):
        if not self.__isinit:
            lock.acquire()
            self.__sess=tf.Session()
            cp = tf.train.latest_checkpoint(self.__model_path)
            save = tf.train.import_meta_graph(cp + '.meta')
            save.restore(self.__sess, cp)
            self.__isinit = True
            lock.release()
        return self.__sess

    '''
    Note:   从文件读取图片并用模型推测验证码
    参数:
        fn: 可以是一个file对象或者是包含路径的文件名字符串
    return: 返回推测的验证码
    '''
    def infer_file(self,fn):
        return self.__do_infer(fn)

    '''
    Note:   从字节对象读取图片并用模型推测验证码
    参数:
        b: b是一个bytes对象
    return: 返回推测的验证码
    '''
    def infer_bytes(self, b):
        return self.__do_infer(BytesIO(b))

    '''
    Note:   从str对象读取图片并用模型推测验证码
    参数:
        string: string是一个str对象
    return: 返回推测的验证码
    '''
    def infer_string(self,string):
        return self.__do_infer(StringIO(string))

    def __do_infer(self,x):
        # session只会初始化一次
        sess=self.__init_session()
        h,w,c,captcha_len,charset_len=sess.run(['height:0','width:0','channel:0','captcha_len:0','charset_len:0'])
        with Image.open(x) as img:
            img_array = np.array(img)
            img = np.reshape(img_array,[1,h*w*c])
            result=sess.run('out:0', feed_dict={'img_p:0': img, 'k_p:0': 1.0})
            result = np.reshape(result, [captcha_len,charset_len])
            return self.__vec2text(result)

class NullDataException(Exception):
    pass
