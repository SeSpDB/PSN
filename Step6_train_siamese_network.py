#! /usr/bin/env python
#coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix #计算混淆矩阵
import re
import os,sys
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from siamese_network_semantic import SiameseLSTMw2v
from tensorflow.contrib import learn
import gzip
from random import random
import gensim
from config import TrainingConfig
from Metrics import Metrics

config = TrainingConfig()
config.Task_name = sys.argv[1] # sys.argv = ['']# 1 定义taskname 2. 定义数据集路径
config.data_path = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/shuf_data/" + config.Task_name + "/"
inpH = InputHelper()
# import wandb
# wandb.init(project="Siamese-test", entity="sec-xd" , name='20_CVE_input_0.8dropout_new')
#
# wandb.config.dropout = 0.8
# wandb.config.hidden_layer_size = 200

# Parameters
# ==================================================


tf.flags.DEFINE_boolean("is_char_based", False, "is character based syntactic similarity. "
                                               "if false then word embedding based semantic similarity is used."
                                               "(default: True)")

# tf.flags.DEFINE_string("word2vec_model", "wiki.simple.vec", "word2vec pre-trained embeddings file (default: None)")
tf.flags.DEFINE_string("word2vec_model", config.word2vec_path, "word2vec pre-trained embeddings file (default: None)")

tf.flags.DEFINE_string("word2vec_format", "bin", "word2vec pre-trained embeddings file format (bin/text/textgz)(default: None)")

tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.9, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularizaion lambda (default: 0.0)")
#tf.flags.DEFINE_string("training_files", "person_match.train2", "training file (default: None)")  #for sentence semantic similarity use "train_snli.txt"
# 数据集读取路径
tf.flags.DEFINE_string("training_files", config.data_path, "training file (default: None)")  #for sentence semantic similarity use "train_snli.txt"
# or /AFTER_TRAIN

tf.flags.DEFINE_integer("hidden_units", 128, "Number of hidden units (default:50)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")#允许动态分配内存
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices") #打印设备信息
max_document_length=512 # 向量的长度

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
inpH.Log.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    inpH.Log.info("{}={}".format(attr.upper(), value))
inpH.Log.info("")

if FLAGS.training_files==None:
    inpH = InputHelper()("Input Files List is empty. use --training_files argument.")
    exit()

# 整理数据集
train_set, dev_set, vocab_processor,sum_no_of_batches = inpH.getDataSets(FLAGS.training_files,max_document_length, 10,FLAGS.batch_size, FLAGS.is_char_based) #10%留作测试集合
trainableEmbeddings=False
if FLAGS.is_char_based==True:
    FLAGS.word2vec_model = False
else:
    if FLAGS.word2vec_model==None:
        trainableEmbeddings=True
        inpH.Log.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
          "You are using word embedding based semantic similarity but "
          "word2vec model path is empty. It is Recommended to use  --word2vec_model  argument. "
          "Otherwise now the code is automatically trying to learn embedding values (may not help in accuracy)"
          "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        inpH.loadW2V(FLAGS.word2vec_model, FLAGS.word2vec_format)

# Training
# ==================================================
inpH.Log.info("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto()
    tf.ConfigProto.allow_soft_placement = FLAGS.allow_soft_placement
    session_conf.log_device_placement = FLAGS.log_device_placement
    sess = tf.Session(config=session_conf)
    inpH.Log.info("started session")
    with sess.as_default():
        if FLAGS.is_char_based:
            siameseModel = SiameseLSTM(
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                hidden_units=FLAGS.hidden_units,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=FLAGS.batch_size
            )
        else:
            siameseModel = SiameseLSTMw2v(
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,# 将每一个token映射到多少维度的向量
                hidden_units=FLAGS.hidden_units,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                batch_size=FLAGS.batch_size,
                trainableEmbeddings=trainableEmbeddings
            )
        # Define Training procedure  #定义训练特征
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=1e-3,
                                           global_step=global_step,
                                           decay_steps=5000,
                                           decay_rate=0.98,
                                           staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate) #定义优化器 adam
        inpH.Log.info("initialized siameseModel object")
    
    grads_and_vars=optimizer.compute_gradients(siameseModel.loss) #计算梯度
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step) #梯度更新
    inpH.Log.info("defined training_ops")
    # Keep track of gradient values and sparsity (optional) #跟踪渐变值和稀疏度
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
            
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    inpH.Log.info("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", config.Task_name +"_" + timestamp))
    inpH.Log.info("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # 保存模型参数 Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    inpH.Log.info("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)

    if FLAGS.word2vec_model :
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        #initW = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # load any vectors from the word2vec
        inpH.Log.info("initializing initW with pre-trained word2vec embeddings")
        for w in vocab_processor.vocabulary_._mapping:
            arr=[]
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in inpH.pre_emb:
                arr=inpH.pre_emb[w]
            elif w.lower() in inpH.pre_emb:
                arr=inpH.pre_emb[w.lower()]
            elif s in inpH.pre_emb:
                arr=inpH.pre_emb[s]
            #elif s.isdigit():
            #    arr=inpH.pre_emb["zero"] # fixme
            if len(arr)>0:
                idx = vocab_processor.vocabulary_.get(w)
                initW[idx]=np.asarray(arr).astype(np.float)
        inpH.Log.info("Done assigning intiW. len="+str(len(initW)))
        inpH.deletePreEmb()
        gc.collect()
        sess.run(siameseModel.W.assign(initW))

    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        feed_dict = 1
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
        _, step, loss, accuracy, dist, sim, summaries = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim, train_summary_op],feed_dict)
        # cm是上一个batch的混淆矩阵，op是更新完当前batch的
        cm, op = _streaming_confusion_matrix(y_batch, sim, 2) # 2为类别总数

        # 将sim结果二值化，用于计算精确度，召回率，和F1分数
        predictions = tf.cast(tf.greater(sim, 0.5), tf.int32)
        precision, recall, f1 = Metrics.calculate_metrics(predictions, y_batch)

        sess.run(tf.local_variables_initializer())  # 初始化所有的local variables
        inpH.Log.info(sess.run(cm))
        inpH.Log.info(sess.run(op))
        cm = op  # 令cm是更新后的混淆矩阵,混淆矩阵的维度和分类总数一致
        # calculate precision
        # pr, re, f1 = inpH.pr_re_f1(cm, [1]) # 1 is positive's category

        # sess.run(tf.local_variables_initializer())

        time_str = datetime.datetime.now().isoformat()
        # print("TRAIN {%s}: step {%s}, loss {%s}, acc {%s}，precision{%s}, recall{%s}, F1{%s} "%(time_str, step, loss, accuracy))
        # inpH.Log.info("TRAIN {%s}: step {%s}, loss {%s}, acc {%s}"%(time_str, step, loss, accuracy))
        inpH.Log.info("Step {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(step, loss, accuracy, precision, recall, f1))
        train_summary_writer.add_summary(summaries, step)
        # print(y_batch, dist, sim) 

    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        feed_dict = 1
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim, summaries = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim, dev_summary_op],  feed_dict)

        cm, op = _streaming_confusion_matrix(y_batch, sim, 2) # 2为类别总数

        sess.run(tf.local_variables_initializer())  # 初始化所有的local variables
        # print(sess.run(cm))
        # print(sess.run(op))
        cm = op  # 令cm是更新后的混淆矩阵,混淆矩阵的维度和分类总数一致
        #calculate precision
        # pr, re, f1 = inpH.pr_re_f1(cm, [1]) #1 is positive's category

        # sess.run(tf.local_variables_initializer())

        time_str = datetime.datetime.now().isoformat()
        # print("DEV {%s}: step {%s}, loss {%s}, acc {%s}，precision{%s}, recall{%s}, F1{%s} "%(time_str, step, loss, accuracy,sess.run(pr),sess.run(re),sess.run(f1)))
        # time_str = datetime.datetime.now().isoformat()

        inpH.Log.info("DEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        # 将sim结果二值化，用于计算精确度，召回率，和F1分数
        predictions = tf.cast(tf.greater(sim, 0.5), tf.int32)
        precision, recall, f1 = Metrics.calculate_metrics(predictions, y_batch)

        # 记录和输出
        inpH.Log.info("Step {}, Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(step, loss, accuracy, precision, recall, f1))
        dev_summary_writer.add_summary(summaries, step)

        #保存运行中间过程 acc、loss等
        dev_summary_writer.add_summary(summaries, step)
        # print(y_batch, sim)
        return accuracy

    # Generate batches
    batches=inpH.batch_iter(
                list(zip(train_set[0], train_set[1], train_set[2])), FLAGS.batch_size, FLAGS.num_epochs)

    # 开始训练
    ptr=0
    max_validation_acc=0.0
    for nn in range(sum_no_of_batches*FLAGS.num_epochs):
        batch = next(batches)
        if len(batch)<1:
            continue
        x1_batch,x2_batch, y_batch = zip(*batch)
        if len(y_batch)<1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc=0.0
        if current_step % FLAGS.evaluate_every == 0: #多少step做一次evaluation
            inpH.Log.info("\nEvaluation:")
            dev_batches = inpH.batch_iter(list(zip(dev_set[0],dev_set[1],dev_set[2])), FLAGS.batch_size, 1)
            for db in dev_batches:
                if len(db)<1:
                    continue
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                if len(y_dev_b)<1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
            inpH.Log.info("")
        if current_step % FLAGS.checkpoint_every == 0: # 做少次存一次checkpoint_every
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                inpH.Log.info("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))

# # record log
# wandb.save("mymodel.h5")
