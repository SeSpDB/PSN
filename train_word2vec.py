#coding=utf-8
#! /usr/bin/env python
from gensim.models.word2vec import Word2Vec
import gensim
import  os


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for line in open(self.dirname):
            yield line.split()


#sentences = MySentences('./train_snli/slice_corpus1.txt')  # a memory-friendly iterator
#model = Word2Vec(sentences,min_count=10,size=200,workers=4) # min_count=szie：删除小于size的单词 size=200：神经网络的层数 workers：多线程数量

#model.save('./word2vec_model/word2vec_model.bin.gz') #存储训练好的模型 不用txt格式是因为：由于以txt文本存储的模型并没有保留训练模型某些重要参数，导致加载该模型时，无法在原先训练好的模型上继续学习

#new_model = gensim.models.Word2Vec.load('./word2vec_model/') #加载模型


# -*- coding: utf-8 -*-
import gensim
import codecs


def main():
    path_to_model = './word2vec_model/word2vecmodel.bin'
    output_file = './word2vec_model/word2vecmodel.txt'
    bin2txt(path_to_model, output_file)


def bin2txt(path_to_model, output_file):
    output = codecs.open(output_file, 'w', 'utf-8')
    model = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True,unicode_errors="ignore")
    print('Done loading Word2Vec!')
    vocab = model.vocab
    for item in vocab:
        vector = list()
        for dimension in model[item]:
            vector.append(str(dimension))
        vector_str = ",".join(vector)
        line = item + "\t" + vector_str
        output.writelines(line + "\n")  # 本来用的是write（）方法，但是结果出来换行效果不对。改成writelines（）方法后还没试过。
    output.close()


if __name__ == "__main__":
    main()