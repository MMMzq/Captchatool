import os
from queue import Queue
from PIL import Image
import numpy as np
import traceback
import tensorflow as tf
import threading
from tensorflow import keras

lock = threading.Lock()
'''
1.不要关注命中率，请把关注力集中到loss是否变小上，
    通常loss，和命中率是负相关的，loss降低一般意味着的命中率的提升！
2.在相同的模型中条件下，模型的大小，实际很大程度取决于图片的大小，和输入到全连接处是shape的大小，
    例如输入到全连接层shape为(batch_size,15,40,1)和(batch_size,8,20,1),后者的模型将比前者小3/2
    其实，输入的图片越大，到全连接层的shape就会越大，
    所以想要模型变小，就要尽量减小输入到全连接层shape的大小，或者取消全连接层。
'''


class Code_tool:
    __label_resolve_func = None
    __input_path = None
    __test_input_path = None
    __out_path = None
    __charset = None
    __batch_size = None
    __filename_queue = None
    __file_queue = None
    __data_queue = None
    __max_captcha_len = None
    __charset_len = None
    __fnl = None
    __isinit=False
    width = None
    height = None
    channel = None
    format = None
    # -----------------------------------------------
    # 利用列表的append,remove的原子操作来在多线程下来计数
    __filename_queue_handle_count = []
    __file_queue_handle_count = []
    __data_queue_handle_count = []

    # 如果只想从已有模型导入且只想用模型推断验证码，input_path,test_input_path可不填
    def __init__(self, charset, out_path, input_path=None, test_input_path=None,
                 label_resolve_func=None, batch_size=32,
                 max_thread_size=16, batch_multiple=4, data_handle_n=3):
        self.__out_path = out_path
        self.__charset = charset
        self.__charset_len = len(charset)
        # 判断是否有路径，如果没有，将不能使用train方法！
        if input_path  and test_input_path :
            self.__input_path = input_path
            self.__test_input_path = test_input_path
            if label_resolve_func is None:
                self.__label_resolve_func = lambda s: s
            else:
                self.__label_resolve_func = label_resolve_func
            self.__batch_size = batch_size
            self.__data_handle_n = data_handle_n
            self.__max_thread_size = max_thread_size
            self.__filename_queue = Queue(self.__batch_size * batch_multiple)
            self.__file_queue = Queue(self.__batch_size * batch_multiple)
            self.__data_queue = Queue(self.__batch_size)
            self.__fnl = os.listdir(self.__input_path)
            fn=self.__input_path + self.__fnl[0]
            img = Image.open(fn)
            self.format=img.format
            # 获取图片形状
            shape = np.array(img).shape
            self.height = shape[0]
            self.width = shape[1]
            if len(shape) == 2:
                self.channel = 1
            else:
                self.channel = shape[2]
            # 获取验证码长度
            self.__max_captcha_len = len(self.__label_resolve_func(self.__fnl[0]))

        '''
       -----------------------启动线程-------------------------
       将有一个文件名队列处理线程，两个组装数据队列处理者线程，其余剩下的线程都是文件队列处理者
       注意对应的处理者请不要超过对应队列的大小
       '''
    def __startwork(self, epochs):
        lock.acquire()
        self.__filename_queue_handle_count.append(None)
        t = threading.Thread(target=self.__filename_queue_handle, daemon=True,
                             args=(self.__fnl, epochs))
        t.start()
        for _ in range(self.__data_handle_n):
            self.__data_queue_handle_count.append(None)
            t = threading.Thread(target=self.__data_queue_handle, daemon=True)
            t.start()
        for _ in range(self.__max_thread_size - self.__data_handle_n - 1):
            self.__file_queue_handle_count.append(None)
            t = threading.Thread(target=self.__file_queue_handle, daemon=True)
            t.start()
        lock.release()

    def __endwork(self):
        self.__file_queue.queue.clear()
        self.__filename_queue.queue.clear()
        self.__data_queue.queue.clear()
        self.__data_queue_handle_count.clear()
        self.__file_queue_handle_count.clear()
        self.__filename_queue_handle_count.clear()

    '''
    ---------------------------训练数据读取处理----------------------
    '''

    def __filename_queue_handle(self, filename_list, epochs):
        try:
            for _ in range(epochs):
                for i in filename_list:
                    # print(1)
                    self.__filename_queue.put(i)
        except BaseException:
            msg = traceback.format_exc()
            print(msg)
            raise BaseException
        finally:
            self.__filename_queue_handle_count.remove(None)
            if len(self.__filename_queue_handle_count) == 0:
                # 全部读取完后，向队列加入None用来告诉消费者已经到末尾了
                self.__filename_queue.put(None)

    # 如果该函数抛出异常，将会导致数据读取不完全！
    def __file_queue_handle(self):
        try:
            while True:
                # print(2)
                file_name = self.__filename_queue.get()
                if file_name is None:
                    # 如果进入该块，表示已经没有数据可读了
                    break
                with Image.open(self.__input_path + file_name) as f:
                    img_array = np.array(f)
                    if img_array.shape[1] != self.width or img_array.shape[0] != self.height:
                        # 格式不符合弃掉
                        continue
                    img_raw = img_array.flatten()
                    fn = self.__label_resolve_func(file_name)
                    label_raw = self.__text2vec(fn)
                    self.__file_queue.put({'img_raw': img_raw, 'label_raw': label_raw}, )
        except BaseException as e:
            msg = traceback.format_exc()
            print(msg)
            raise BaseException
        finally:
            self.__file_queue_handle_count.remove(None)
            if len(self.__file_queue_handle_count) == 0:
                self.__file_queue.put(None)  # 发出数据处理者线程该结束的信号
            else:
                self.__filename_queue.put(None)  # 通知别的文件处理者线程结束的线程结束

    # 如果该函数抛出异常，将会导致数据读取不完全！
    def __data_queue_handle(self):
        try:
            while True:
                # print(3)
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
                self.__data_queue.put(None)
            else:
                self.__file_queue.put(None)

    def __do_add_data_queue(self, batch_img, batch_label):
        batch_label = np.reshape(batch_label, [-1, self.__max_captcha_len * self.__charset_len])
        batch_img = np.reshape(batch_img, [-1, self.height * self.width * self.channel])
        self.__data_queue.put({'batch_img': batch_img, 'batch_label': batch_label}, )

    # 该方法不一定会返回等同与batch_size参数的batch,例如最后一批可能只有10张图片组成
    def generate_next_batch(self):
        data = self.__data_queue.get()
        if data == None:
            raise NullDataException('空数据，原因所有数据已被读取')
        return data['batch_img'], data['batch_label']

    # 重要:test文件夹和train文件夹一定不要相同！！！否则会影响测试命中率,而且测试集不宜太大，太多可能会导致OOM
    def generate_test_batch(self):
        batch_img = np.array([])
        batch_label = np.array([])

        for fn in os.listdir(self.__test_input_path):
            with Image.open(self.__test_input_path + fn) as f:
                img_array = np.array(f)
                if img_array.shape[1] != self.width or img_array.shape[0] != self.height:
                    # 格式不符合弃掉
                    continue
                img_raw = img_array.flatten()
                fn = self.__label_resolve_func(fn)
                label_raw = self.__text2vec(fn)
                batch_img = np.append(batch_img, img_raw)
                batch_label = np.append(batch_label, label_raw)

        batch_label = np.reshape(batch_label, [-1, self.__max_captcha_len * self.__charset_len])
        batch_img = np.reshape(batch_img, [-1, self.height * self.width * self.channel])
        return batch_img, batch_label

    def __text2vec(self, text):
        vec = np.zeros([self.__max_captcha_len, self.__charset_len])
        for i in range(self.__max_captcha_len):
            index = self.__charset.find(text[i])
            vec[i, index] = 1
        vec = vec.flatten()
        return vec

    def __vec2text(self, vec):
        # vec.shape 要等于[captcha_len,charset_len]，否则会不正确
        indexs = vec.argmax(axis=1)
        text = ''
        for i in indexs:
            text += self.__charset[i]
        return text

    '''
    ---------------------cnn定义--------------------------
    
    '''
    def init_cnn(self):
        # 定义占位符，变量,常量
        #这里的常量用于以后推测验证码时使用
        tf.constant(self.height,name='height')
        tf.constant(self.width,name='width')
        tf.constant(self.channel,name='channel')
        tf.constant(self.__max_captcha_len,name='captcha_len')
        tf.constant(self.__charset_len,name='charset_len')
        tf.constant(self.format,name='format',dtype=tf.string)
        img_b = tf.placeholder(tf.float32, [None, self.height * self.width * self.channel],name='i_p')
        label_b = tf.placeholder(tf.float32, name='l_p')
        keed = tf.placeholder(tf.float32, name='k_p')
        global_step = tf.Variable(0, name='g_v')
        img_b = tf.reshape(img_b, shape=[-1, self.height, self.width, self.channel])
        w_alpha =0.01
        b_alpha =0.1

        def conv2(input, ksize, padding='SAME'):
            w = tf.Variable(w_alpha * tf.random_normal(ksize))
            b = tf.Variable(b_alpha * tf.random_normal([ksize[3]]))
            return tf.nn.bias_add(tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding=padding), b)
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        h = conv2(img_b, [3, 3, self.channel, 32])
        h = tf.nn.relu(h)
        h = max_pool_2x2(h)
        h = tf.nn.dropout(h, keed)

        h = conv2(h, [3, 3, 32, 64])
        h = tf.nn.relu(h)
        h = max_pool_2x2(h)
        h = tf.nn.dropout(h, keed)

        h = conv2(h, [3, 3, 64, 64])
        h = tf.nn.relu(h)
        h = max_pool_2x2(h)
        h = tf.nn.dropout(h, keed)
        print(h.get_shape())

        shape = h.get_shape().as_list()
        w_f1 = tf.Variable(w_alpha * tf.truncated_normal([shape[1] * shape[2] * shape[3], 1024]))
        b_f1 = tf.Variable(b_alpha * tf.truncated_normal([1024]))
        dense = tf.reshape(h, [-1, w_f1.get_shape().as_list()[0]])
        dense = tf.add(tf.matmul(dense, w_f1), b_f1)
        dense = tf.nn.relu(dense)
        dense = tf.nn.dropout(dense, keed)
        print(dense.get_shape())

        w_out = tf.Variable(
            w_alpha * tf.truncated_normal([1024, self.__max_captcha_len * self.__charset_len]))
        b_out = tf.Variable(
            b_alpha * tf.truncated_normal([self.__max_captcha_len * self.__charset_len]))
        out = tf.add(tf.matmul(dense, w_out), b_out, 'out')
        print(out.get_shape())

        #   定义操作
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_b, logits=out),
                              name='loss')
        train_op = tf.train.AdamOptimizer().minimize(loss, name='train')
        predict = tf.reshape(out, [-1, self.__max_captcha_len, self.__charset_len])
        max_logits_indexs = tf.argmax(predict, 2, 'max')
        label_b = tf.reshape(label_b, [-1, self.__max_captcha_len, self.__charset_len])
        max_label_indexs = tf.argmax(label_b, 2)
        correct_pred = tf.equal(max_label_indexs, max_logits_indexs)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


    # 该方法请不要在多线程环境运行,请确保同一时间内只有一个线程运行该方法
    def train(self, epochs, target_ac=1., retrain=True):
        if self.__input_path is None or self.__test_input_path is None:
            print('__input_path或__test_input_path缺失')
            return
        self.__startwork(epochs)
        #   开始训练
        with tf.Session() as sess:
            if retrain:
                self.init_cnn()
                save = tf.train.Saver()
                sess.run(tf.global_variables_initializer())
                step = 0
            else:
                cp = tf.train.latest_checkpoint(self.__out_path)
                save = tf.train.import_meta_graph(cp + '.meta')
                save.restore(sess, cp)
                step = int(sess.run('g_v:0'))
            test_img, test_label = self.generate_test_batch()
            while True:
                try:
                    img, label = self.generate_next_batch()
                except NullDataException:
                    break
                loss, _ = sess.run(['loss:0', 'train'],
                                   feed_dict={'i_p:0': img, 'l_p:0': label, 'k_p:0': 1})
                print('步数为：{}\tloss:{}'.format(step, loss))
                if step % 10 == 0:
                    # 这里的命中率不代表真正的命中率，例如，模型推测出某个标签是1234，但正确的标签是1235，按照常理来讲这里并不相等，但是对于这里命中率来说，命中率是百分之75.也就是说这里的命中是针对单个字符的
                    actual_ac = sess.run('accuracy:0',
                                         feed_dict={'i_p:0': test_img, 'l_p:0': test_label,
                                                    'k_p:0': 1.})
                    print('步数为：{}\t命中率:{}'.format(step, actual_ac))
                    if actual_ac >= target_ac:
                        break
                step += 1
            # 保存
            g_step = tf.get_default_graph().get_tensor_by_name('g_v:0')
            tf.assign(g_step, step, name='update')
            sess.run('update:0')
            self.__endwork()
            print('保存模型中请等待!')
            save.save(sess, self.__out_path + 'model', global_step=step)
            print('完成')


    def __init_session(self):
        if not self.__isinit:
            lock.acquire()
            self.sess=tf.Session()
            cp = tf.train.latest_checkpoint(self.__out_path)
            save = tf.train.import_meta_graph(cp + '.meta')
            save.restore(self.sess, cp)
            self.__isinit = True
            lock.release()
        return self.sess

    def infer(self,fn):
        sess=self.__init_session()
        h,w,c,captcha_len,charset_len=sess.run(['height:0','width:0','channel:0','captcha_len:0','charset_len:0'])
        with Image.open(fn) as f:
            img_array = np.array(f)
            img = np.reshape(img_array,[1,h*w*c])
            result=sess.run('out:0', feed_dict={'i_p:0': img, 'k_p:0': 1.0})
            result = np.reshape(result, [captcha_len,charset_len])
            return self.__vec2text(result)

    def infet_bytes(self, bytes):
        sess=self.__init_session()
        h,w,c,captcha_len,charset_len,format=sess.run(['height:0','width:0','channel:0','captcha_len:0','charset_len:0','format:0'])
        # 这里format是一个bytes对象所以要转换
        format=str(format,encoding='utf-8')
        if format =='PNG':
            decode=tf.image.decode_png(bytes, c)
        elif format == 'JPEG':
            decode=tf.image.decode_jpeg(bytes,c)
        else:
            raise  TypeError('不支持的格式,请切换别的方法')
        result=sess.run(decode)
        img_array = np.array(result)
        img = np.reshape(img_array,[1,h*w*c])
        result=sess.run('out:0', feed_dict={'i_p:0': img, 'k_p:0': 1.0})
        result = np.reshape(result, [captcha_len,charset_len])
        return self.__vec2text(result)

class NullDataException(Exception):
    pass
