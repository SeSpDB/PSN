#!/usr/bin/env python
# -*- coding: utf-8 -*-
class TrainingConfig:
    def __init__(self):

        self.Task_name = "slice1215"
        self.root_dir = "/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/"
        self.voc_path = './vocabulary/vocabulary.pkl'
        # self.word2vec_path = './vocabulary/word2vec.bin'
        self.word2vec_path = '/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/word2vec_model/own_word2vec_model.bin.gz'
        self.use_word2vec = 1
        # self.data_path = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch'  #./train_snli/AST_TRAIN
        self.data_path = "/home/deeplearning/nas-files/tracer/data/equal_patch_data/train_pair_patch/slice1215/"
        self.Copy_path = '/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Evaluation/RQ3 Abalation Evaluation/slice1215.csv'
        self.slice_path = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice1215'
        self.No_slice_path = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/slicing_patch/slice1215_none/' 
        self.SYS_FUNC_MAPPING_PATH = self.root_dir +"/data_process/function.xls"
        self.Synthetic_path = '/home/deeplearning/nas-files/tracer/data/equal_patch_data/synthetic_patch/'
        self.Target_path = "/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Evaluation/RQ3 Abalation Evaluation/"
        self.Voc_path = '/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/vocabulary/vocab_processor.bin'
        # Log
        self.Log_save_path = "/home/deeplearning/nas-files/tracer/src/tracer/tracer-master/siamese-lstm-gpu/Logs/"

# 使用示例
args = TrainingConfig()