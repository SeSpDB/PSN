#coding=utf-8
from boto import config
import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys,os
import psutil
import tensorflow as tf
from config import TrainingConfig
Args = TrainingConfig()
from Logs import Log_writer
#reload(sys)
#sys.setdefaultencoding("utf-8")


class MySentences(object): #利用yield方法读取数据集，防止因为数据量太大，ram溢出
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class InputHelper(object):
    pre_emb = dict()
    vocab_processor = None

    def __init__(self):
        timestamp = str(int(time.time()))
        self.Log = Log_writer.get_logger(Args.Log_save_path + Args.Task_name + timestamp) # 加载日志函数

    def cleanText(self, s):
        s = re.sub(r"[^\x00-\x7F]+"," ", s)
        s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+',"",s)
        s = re.sub(r'( [0-9,\.]+)',r"\1 ", s)
        s = re.sub(r'\$'," $ ", s)
        s = re.sub('[ ]+',' ', s)
        return s.lower()
        
    def getVocab(self,vocab_path, max_document_length,filter_h_pad):
        if self.vocab_processor==None:
            print('locading vocab')
            vocab_processor = MyVocabularyProcessor(max_document_length-filter_h_pad,min_frequency=0)
            self.vocab_processor = vocab_processor.restore(vocab_path)
        return self.vocab_processor

    def loadW2V(self,emb_path, type="bin"):
        print("Loading W2V data...")
        num_keys = 0
        if type=="textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        if type=="text":
            # this seems faster than gensim non-binary load
            for line in open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        else:
            # pre_emd是一个Word2Vec类
            self.pre_emb = KeyedVectors.load(emb_path) #使用“ignore”忽略 utf-8产生的编码错误  load_word2vec_format(emb_path,binary=True,unicode_errors='ignore')
            # init_sims  进行归一化
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.wv.vocab) # fixme
        print("loaded word2vec len ", num_keys)     #num_keys = 1912
        gc.collect()

    def deletePreEmb(self):
        self.pre_emb=dict()
        gc.collect()

        #创建读取孪生数据集
    # def vice_getTsvData(self,filepath):
    #     for every_txt in os.listdir(filepath):
    #         input_path = os.path.join(filepath,every_txt)
    #         f = open(input_path, 'r')
    #         for line in open(input_path):
    #             l = line.strip().split("???")  # 修改分割符
    #             if len(l) < 2:
    #                 continue
    #             if random() > 0.5:
    #                 yield (l[0].lower(), l[1].lower(),l[2])
    #             else:
    #                 yield (l[1].lower(), l[0].lower(),l[2])
    #         f.close()

    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        mem = psutil.virtual_memory()
        # 系统总内存
        total_mem = float(mem.total)/1024/1024/1024
        # 已使用内存
        used_mem = float(mem.used)/1024/1024/1024
        # 未使用内存
        free_mem = float(mem.free)/1024/1024/1024
        print('系统总内存：%d GB' %total_mem )
        print('系统已使用内存:%d GB' %used_mem)
        print('系统剩余内存：%d GB' %free_mem)
        for every_txt in os.listdir(filepath):
            all_line = []
            input_path = os.path.join(filepath,every_txt)
            with open(input_path,'r') as f:
                while True:
                    line = f.readline()
                    line = line.strip()
                    all_line.append(line)
                    if not line:
                        break
            for every in all_line:
                l = every.strip().split("???")  # 修改分割符
                if len(l) < 2:
                    continue
                if len(l[2])>4:
                    continue

                try:
                    # 尝试将第三个字段转换为整数
                    label = int(l[2])
                except ValueError:
                    # 如果转换失败，打印错误和对应的行内容
                    print("Cannot convert to int:", l)
                    continue  # 跳过这行数据
                
                if random() > 0.5:
                    x1.append(l[0].lower())
                    x2.append(l[1].lower())
                else:
                    x1.append(l[1].lower())
                    x2.append(l[0].lower())
                y.append(label)

        print('now total...')
        size_x1 = sys.getsizeof(x1) / 1024 / 1024
        size_x2 = sys.getsizeof(x2) / 1024 / 1024
        size_y = sys.getsizeof(y) / 1024 / 1024
        print('the size of x1 is MB',size_x1)
        print('the size of x2 is MB', size_x2)
        print('the size of y is MB', size_y)
        temp3 = np.asarray(y,dtype=bytearray)
        mem = psutil.virtual_memory()
        # 系统总内存
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        # 已使用内存
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        # 未使用内存
        free_mem = float(mem.free) / 1024 / 1024 / 1024
        print('系统总内存：%d GB' % total_mem)
        print('系统已使用内存:%d GB' % used_mem)
        print('系统剩余?内存：%d GB' % free_mem)

        temp1 = np.array(x1,dtype=bytearray)
        mem = psutil.virtual_memory()
        # 系统总内存
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        # 已使用内存
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        # 未使用内存
        free_mem = float(mem.free) / 1024 / 1024 / 1024
        print('系统总内存：%d GB' % total_mem)
        print('系统已使用内存:%d GB' % used_mem)
        print('系统剩余?内存：%d GB' % free_mem)
        temp2 = np.asarray(x2,dtype=bytearray)
        mem = psutil.virtual_memory()
        # 系统总内存
        total_mem = float(mem.total) / 1024 / 1024 / 1024
        # 已使用内存
        used_mem = float(mem.used) / 1024 / 1024 / 1024
        # 未使用内存
        free_mem = float(mem.free) / 1024 / 1024 / 1024
        print('系统总内存：%d GB' % total_mem)
        print('系统已使用内存:%d GB' % used_mem)
        print('系统剩余?内存：%d GB' % free_mem)

        return temp1,temp2,temp3
        # return np.asarray(x1),np.asarray(x2),np.asarray(y)

    def getTsvDataCharBased(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        for every_txt in os.listdir(filepath):
            input_path = os.path.join(filepath,every_txt)
            # positive samples from file
            for line in open(input_path):
                l=line.strip().split("???")#strip方法去除头部、尾部的字符
                if len(l)<2:
                    continue
                if random() > 0.5:
                   x1.append(l[0].lower())
                   x2.append(l[1].lower())
                else:
                   x1.append(l[1].lower())
                   x2.append(l[0].lower())
                y.append(1)#np.array([0,1]))
            # generate random negative samples
            combined = np.asarray(x1+x2)
            shuffle_indices = np.random.permutation(np.arange(len(combined)))
            combined_shuff = combined[shuffle_indices]
            for i in xrange(len(combined)):
                x1.append(combined[i])
                x2.append(combined_shuff[i])
                y.append(0) #np.array([1,0]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        for every_txt in os.listdir(filepath):
            input_path = os.path.join(filepath,every_txt)
            # positive samples from file
            for line in open(input_path):
                l=line.strip().split("\t")
                if len(l)<3:
                    continue
                x1.append(l[1].lower())
                x2.append(l[2].lower())
                y.append(int(l[0])) #np.array([0,1]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)  
 
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        # print(data)
        # print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                # shuffle_indices为一个从0到data_size的随机数组,例如data_size为4时，为(3  0  1  2)
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i),'w') as f:
            for text1,text2,label in zip(x1_dev,x2_dev,y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size, is_char_based):
        if is_char_based:
            x1_text, x2_text, y=self.getTsvDataCharBased(training_paths)
        else:
            # x1  x2分别是切片对，y为该切片对的标签
            x1_text, x2_text, y=self.getTsvData(training_paths)
            # 检查是否存在已保存的 VocabularyProcessor
        vocab_processor_path = Args.Voc_path 
        if os.path.exists(vocab_processor_path) and Args.use_word2vec == 1:  #  如果存在词向量 或 确定使用 就使用词向量模型
            print("Loading existing vocabulary processor")
            vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=is_char_based)
            vocab_processor = vocab_processor.restore(vocab_processor_path)
        else:
            print("Building and saving vocabulary")
            vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=is_char_based)
            vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
            vocab_processor.save(vocab_processor_path)

        # 1.Build vocabulary
        # vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=is_char_based)
        # vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        # vocab_processor.save(vocab_processor_path)

        print("Length of loaded vocabulary ={}".format( len(vocab_processor.vocabulary_)))      #  test是4439
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))

        # 2.Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y))) # 对所有数据进行随机排列、仅限于一维，多维数据也仅限于一维
        x1_shuffled = x1[shuffle_indices]       #  将x1  x2  y都打乱
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100 # //表示整数除法      训练集  测试集大概9比1
        del x1
        del x2

        # 3.Split train/test set
        self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,0)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)     #  统计一次需要多少个batchs
        train_set=(x1_train,x2_train,y_train)
        dev_set=(x1_dev,x2_dev,y_dev)
        gc.collect()         #回收内存
        return train_set,dev_set,vocab_processor,sum_no_of_batches
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp,x2_temp,y = self.getTsvTestData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print(len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1,x2, y

