import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from PIL import Image
import numpy as np
import traceback
import tensorflow as tf
import threading

lock=threading.Lock()

class Code_tool:
    __label_resolve_func = None
    __input_path = None
    __test_input_path=None
    __out_path = None
    __charset = None
    __batch_size = None
    __filename_queue = None
    __file_queue = None
    __data_queue = None
    __threadpool = None
    __max_captcha_len = None
    __charset_len = None
    __training=None
    width=None
    height=None
    channel=None
    # -----------------------------------------------
    # 利用列表的append,remove的原子操作来在多线程下来计数
    __filename_queue_handle_count = []
    __file_queue_handle_count = []
    __data_queue_handle_count = []
    #
    def __init__(self, input_path,test_input_path, out_path, charset, label_resolve_func=None, batch_size=32,
                 max_thread_size=16, batch_multiple=4):
        if label_resolve_func is None:
            self.__label_resolve_func = lambda s: s
        else:
            self.__label_resolve_func = label_resolve_func
        self.__input_path = input_path
        self.__test_input_path=test_input_path
        self.__out_path = out_path
        self.__charset = charset
        self.__charset_len = len(charset)
        self.__training=False
        self.__batch_size = batch_size
        self.__max_thread_size = max_thread_size
        self.__filename_queue = Queue(self.__batch_size * batch_multiple)
        self.__file_queue = Queue(self.__batch_size * batch_multiple)
        self.__data_queue = Queue(self.__batch_size*batch_multiple)
        self.__threadpool = ThreadPoolExecutor(self.__max_thread_size)

    def __startWork(self,filename_list,epochs):
        img = Image.open(self.__input_path + filename_list[0])
        # 获取图片形状
        shape = np.array(img).shape
        self.height=shape[0]
        self.width=shape[1]
        if len(shape) == 2:
            self.channel=1
        else:
            self.channel=shape[2]

        # 获取验证码长度
        self.__max_captcha_len = len(self.__label_resolve_func(filename_list[0]))
        '''
        -----------------------启动线程-------------------------
        将有一个文件名队列处理线程，两个组装数据队列处理者线程，其余剩下的线程都是文件队列处理者
        注意对应的处理者请不要超过对应队列的大小
        '''
        lock.acquire()
        self.__filename_queue_handle_count.append(None)
        self.__threadpool.submit(self.__filename_queue_handle, filename_list,epochs)
        for _ in range(3):
            self.__data_queue_handle_count.append(None)
            self.__threadpool.submit(self.__data_queue_handle)
        for _ in range(self.__max_thread_size - 4):
            self.__file_queue_handle_count.append(None)
            self.__threadpool.submit(self.__file_queue_handle)
        lock.release()

    '''
    ---------------------------训练数据读取处理----------------------
    '''

    def __filename_queue_handle(self, filename_list,epochs):
        try:
            for _ in range(epochs):
                for i in filename_list:
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

    #如果该函数抛出异常，将会导致数据读取不完全！
    def __file_queue_handle(self):
        try:
            while True:
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
                    self.__file_queue.put({'img_raw': img_raw, 'label_raw': label_raw})
        except BaseException as e:
            msg = traceback.format_exc()
            # print(msg)
            raise BaseException
        finally:
            self.__file_queue_handle_count.remove(None)
            if len(self.__file_queue_handle_count) == 0:
                self.__file_queue.put(None)  # 发出数据处理者线程该结束的信号
            else:
                self.__filename_queue.put(None)  # 通知别的文件处理者线程结束的线程结束

    #如果该函数抛出异常，将会导致数据读取不完全！
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
                self.__data_queue.put(None)
            else:
                self.__file_queue.put(None)

    def __do_add_data_queue(self, batch_img, batch_label):
        batch_label = np.reshape(batch_label, [-1, self.__max_captcha_len, self.__charset_len])
        batch_img = np.reshape(batch_img, [-1,self.height,self.width,self.channel])
        self.__data_queue.put({'batch_img': batch_img, 'batch_label': batch_label})

    # 该方法不一定会返回等同与batch_size参数的batch,例如最后一批可能只有10张图片组成
    def __generate_next_batch(self):
        data = self.__data_queue.get()
        if data == None:
            raise NullDataException('空数据，原因所有数据已被读取')
        return data['batch_img'], data['batch_label']

    # 重要:test文件夹和train文件夹一定不要相同！！！否则会影响测试命中率
    def __generate_test_batch(self):
        batch_img = np.array([])
        batch_label = np.array([])

        for fn in os.listdir(self.__test_input_path):
            with Image.open(self.__test_input_path+fn) as f:
                img_array = np.array(f)
                if img_array.shape[1] != self.width or img_array.shape[0] != self.height:
                    # 格式不符合弃掉
                    continue
                img_raw = img_array.flatten()
                fn = self.__label_resolve_func(fn)
                label_raw = self.__text2vec(fn)
                batch_img = np.append(batch_img, img_raw)
                batch_label = np.append(batch_label,label_raw)

        batch_img = np.reshape(batch_img, [-1,self.height,self.width,self.channel])
        batch_label = np.reshape(batch_label, [-1, self.__max_captcha_len, self.__charset_len])
        return batch_img,batch_label



    def __text2vec(self, text):
        vec = np.zeros([self.__max_captcha_len, self.__charset_len])
        for i in range(self.__max_captcha_len):
            index = self.__charset.find(text[i])
            vec[i, index] = 1
        return vec.flatten()

    def __vec2text(self, vec):
        vec = np.reshape(vec, [self.__max_captcha_len, self.__charset_len])
        indexs = vec.argmax(axis=1)
        text = ''
        for i in indexs:
            text += self.__charset[i]
        return text

    '''
    ---------------------cnn定义--------------------------
    '''

    def def_cnn(self,img_b,keep):
        # shape=(batch_size,height,width,channel)
        # img_b=tf.reshape(img_b,[-1,self.height,self.width,self.channel])

        w_y1=tf.Variable(tf.truncated_normal([5,5,self.channel,32]))
        b_y1=tf.Variable(tf.truncated_normal([32]))
        conv_y1=self.con2d(img_b,w_y1,b_y1)
        conv_y1=tf.nn.relu(conv_y1)
        conv_y1=self.max_pool_2x2(conv_y1)
        print(conv_y1.get_shape())

        w_y2=tf.Variable(tf.truncated_normal([5,5,32,64]))
        b_y2=tf.Variable(tf.truncated_normal([64]))
        conv_y2=self.con2d(conv_y1,w_y2,b_y2)
        conv_y2=tf.nn.relu(conv_y2)
        conv_y2=self.max_pool_2x2(conv_y2)
        print(conv_y2.get_shape())

        #shape=(batch_size,height/4,width/4,64,in_channel_size)
        shape=conv_y2.get_shape().as_list()
        w_f1=tf.Variable(tf.truncated_normal([shape[1]*shape[2]*shape[3],1024]))
        b_f1=tf.Variable(tf.truncated_normal([1024]))
        dense=tf.reshape(conv_y2,[-1,w_f1.get_shape().as_list()[0]])
        dense=tf.add(tf.matmul(dense,w_f1),b_f1)
        dense=tf.nn.relu(dense)
        # dense=tf.nn.dropout(dense,keep)
        print(dense.get_shape())
        # dense.shape=(height*width*channel,1024)

        w_out=tf.Variable(tf.truncated_normal([1024,self.__max_captcha_len*self.__charset_len]))
        b_out=tf.Variable(tf.truncated_normal([self.__max_captcha_len*self.__charset_len]))
        out=tf.add(tf.matmul(dense,w_out),b_out)
        print(out.get_shape())
        out=tf.reshape(out,[-1, self.__max_captcha_len, self.__charset_len])
        print(out.get_shape())
        return out

    def con2d(self,x,w,b):
        return tf.nn.bias_add(tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME'),b)


    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def train(self,epochs=10):
        #判断是否有模型训练中
        if self.__training:
            print('已有训练的任务，请等待该任务后结束后再调用')
            return
        #   初始化工作
        self.__training=True
        fl=os.listdir(self.__input_path)
        self.__startWork(fl,epochs)
        img_b=tf.placeholder(tf.float32,[None,self.height,self.width,self.channel])
        label_b=tf.placeholder(tf.float32)
        keed=tf.placeholder(tf.float32)
        step=0
        #   定义操作
        out=self.def_cnn(img_b,keed)
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_b,logits=out))
        train_op=tf.train.AdamOptimizer().minimize(loss)
        max_logits_indexs=tf.argmax(out,2)
        max_label_indexs=tf.argmax(label_b,2)
        correct_pred = tf.equal(max_label_indexs, max_logits_indexs)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        #   开始训练
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            test_img,test_label=self.__generate_test_batch()
            while True:
                try:
                    img,label=self.__generate_next_batch()
                except NullDataException:
                        break
                op=sess.run([loss,train_op],feed_dict={img_b:img,label_b:label,keed:0.75})
                print('步数为：{}\tloss:{}'.format(step,op[0]))
                if step%10==0:
                    actual_ac=sess.run([accuracy],feed_dict={img_b:test_img,label_b:test_label,keed:1.})
                    print('步数为：{}\t命中率:{}'.format(step,actual_ac))
                step+=1
        self.__training=False


    def infer(self, img):
        pass


class NullDataException(Exception):
    pass
