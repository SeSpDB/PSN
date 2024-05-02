#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

class SiameseLSTMw2v(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    
    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden=hidden_units
        n_layers=5
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2


    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, trainableEmbeddings):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings,name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)
        print(self.embedded_words1)
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.out1=self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units)
            self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # with tf.name_scope("precision"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
        #                                 name="temp_sim")  # auto threshold 0.5
        #     print(self.temp_sim,self.input_y)
        #     tf.Print(self.temp_sim, [self.temp_sim])
        #     correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #     # input_y  类型  typle  y_batch = (1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
        #     TP = 0
        #     TN = 0
        #     FP = 0
        #     FN = 0
        #     # 一个input_y为一个64位的tuple
        #     for i in range(64):
        #         if self.input_y == 1 and correct_predictions[i] == True:
        #             TP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == True:
        #             TN += 1
        #         elif self.input_y == 1 and correct_predictions[i] == False:
        #             FP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == False:
        #             FN += 1
        #     self.precision = self.safe_div(TP + TN, TP + FP + TN + FN)


        # with tf.name_scope("precision"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance),name= "temp_sim")
        #     #计算混淆矩阵
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer()) # 变量初始化
        #     cm, op = _streaming_confusion_matrix(self.input_y, self.temp_sim, num_classes = 2) #类别暂时定义为2
        #     print("cm and op",cm,op)
        #     cm = op  # 令cm是更新后的混淆矩阵
        #
        #     pr, re, f1 = self.pr_re_f1(cm, [1]) #1代表是正样本的标签
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer()) # 变量初始化
        #     self.precision = pr

        # with tf.name_scope("recall"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name="temp_sim")
        #     # 计算混淆矩阵
        #     with tf.Session() as sess:
        #         sess.run(tf.global_variables_initializer()) # 变量初始化
        #     cm, op = _streaming_confusion_matrix(self.input_y, self.temp_sim, num_classes=2)  # 类别暂时定义为2
        #
        #     cm = op  # 令cm是更新后的混淆矩阵
        #
        #     pr, re, f1 = self.pr_re_f1(cm, [1])  # 1代表是正样本的标签
        #     with tf.Session() as sess:
        #         sess.run(tf.local_variables_initializer())
        #     self.recall = re
        #
        #
        # with tf.name_scope("F1"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance),name= "temp_sim")
        #     #计算混淆矩阵
        #     cm, op = _streaming_confusion_matrix(self.input_y, self.temp_sim, num_classes = 2) #类别暂时定义为2
        #
        #     cm = op  # 令cm是更新后的混淆矩阵
        #
        #     pr, re, f1 = self.pr_re_f1(cm, [1]) #1代表是正样本的标签
        #     with tf.Session() as sess:
        #         sess.run(tf.local_variables_initializer())
        #     self.f1 = f1

        ###  PKF  add code   ###

        # with tf.name_scope("precision"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
        #                                 name="temp_sim")  # auto threshold 0.5
        #     correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #     # input_y  类型  typle  y_batch = (1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
        #     TP = 0
        #     TN = 0
        #     FP = 0
        #     FN = 0
        #     # 一个input_y为一个64位的tuple
        #     for i in range(64):
        #         if self.input_y == 1 and correct_predictions[i] == True:
        #             TP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == True:
        #             TN += 1
        #         elif self.input_y == 1 and correct_predictions[i] == False:
        #             FP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == False:
        #             FN += 1
        #     self.precision = (TP + TN) / (TP + FP + TN + FN)
        #
        # with tf.name_scope("recall"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
        #                                 name="temp_sim")  # auto threshold 0.5
        #     correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #     # input_y  类型  typle  y_batch = (1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
        #     TP = 0
        #     TN = 0
        #     FP = 0
        #     FN = 0
        #     precision = 0
        #     # 一个input_y为一个64位的tuple
        #     for i in range(64):
        #         if self.input_y == 1 and correct_predictions[i] == True:
        #             TP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == True:
        #             TN += 1
        #         elif self.input_y == 1 and correct_predictions[i] == False:
        #             FP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == False:
        #             FN += 1
        #     self.recall = TP / (TP + FN)
        #
        # with tf.name_scope("FP"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
        #                                 name="temp_sim")  # auto threshold 0.5
        #     correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #     # input_y  类型  typle  y_batch = (1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
        #     TP = 0
        #     TN = 0
        #     FP = 0
        #     FN = 0
        #     precision = 0
        #     # 一个input_y为一个64位的tuple
        #     for i in range(64):
        #         if self.input_y == 1 and correct_predictions[i] == True:
        #             TP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == True:
        #             TN += 1
        #         elif self.input_y == 1 and correct_predictions[i] == False:
        #             FP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == False:
        #             FN += 1
        #     self.FP = FP / 64
        #
        # with tf.name_scope("FN"):
        #     self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
        #                                 name="temp_sim")  # auto threshold 0.5
        #     correct_predictions = tf.equal(self.temp_sim, self.input_y)
        #     # input_y  类型  typle  y_batch = (1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0)
        #     TP = 0
        #     TN = 0
        #     FP = 0
        #     FN = 0
        #     precision = 0
        #     # 一个input_y为一个64位的tuple
        #     for i in range(64):
        #         if self.input_y == 1 and correct_predictions[i] == True:
        #             TP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == True:
        #             TN += 1
        #         elif self.input_y == 1 and correct_predictions[i] == False:
        #             FP += 1
        #         elif self.input_y == 0 and correct_predictions[i] == False:
        #             FN += 1
        #     self.FN = FN / 64
