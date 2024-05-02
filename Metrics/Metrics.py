#! /usr/bin/env python
#coding=utf-8
import tensorflow as tf
from tensorflow.python.ops import metrics as metrics_lib

def calculate_metrics(predictions, labels):
    # 定义评估指标
    precision = metrics_lib.precision(labels, predictions)
    recall = metrics_lib.recall(labels, predictions)
    precision = precision[0]  # Access the first element of the precision tuple
    recall = recall[0]  # Access the first element of the recall tuple
    f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 添加小数避免除以零
    return precision, recall, f1