# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import six
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
from tensorflow.contrib import learn  # pylint: disable=g-bad-import-order

# 分词，一段中有除了数字和字母以外的其他字符，则以这些字符为分隔符
TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)

def tokenizer_char(iterator):
    for value in iterator:
        yield list(value)

def tokenizer_word(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value)
    
class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
               max_document_length,
               min_frequency=0,
               vocabulary=None,
               is_char_based=True):
        if is_char_based:
            tokenizer_fn=tokenizer_char
        else:
            tokenizer_fn=tokenizer_word
        sup = super(MyVocabularyProcessor,self)
        sup.__init__(max_document_length,min_frequency,vocabulary,tokenizer_fn)


    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length: # 截断
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids

    def save(self, filename):
        """Saves vocabulary processor into given file."""
        super(MyVocabularyProcessor, self).save(filename)

    @classmethod
    def restore(cls, filename):
        """Restores vocabulary processor from given file."""
        return super(MyVocabularyProcessor, MyVocabularyProcessor).restore(filename)
