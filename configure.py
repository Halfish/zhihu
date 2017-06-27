from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import os


def readFrom(filename):
    filepath = os.path.join('./ieee_zhihu_cup', filename)
    with open(filepath, 'r') as f:
        return [line.strip('\n').split('\t') for line in f.readlines()]


class GlobalVariable(object):
    '''
    modularize this class to avoid time-consuming data loading
    '''
    def __init__(self):
        print 'loading data...'
        self.char_vectors = KeyedVectors.load_word2vec_format( \
                './ieee_zhihu_cup/char_embedding.txt', binary=False)
        self.word_vectors = KeyedVectors.load_word2vec_format( \
                './ieee_zhihu_cup/word_embedding.txt', binary=False)
        self.topics = readFrom('topic_info.txt')
        self.question_eval = readFrom('question_eval_set.txt')
        self.question_train = readFrom('question_train_set.txt')
        self.question_train_topic = readFrom('question_topic_train_set.txt')
        print 'end of loading'
        
        self.n_classes = len(self.topics)
        self.index2topic = [topic[0] for topic in self.topics]
        self.topic2index = {self.topics[i][0] : i for i in range(self.n_classes)}
        self.class_weights = self._get_class_weights()
        print 'end of init'

    def _get_class_weights(self):
        topic_count = dict((topic[0], 0) for topic in self.topics)
        for topic in self.question_train_topic:
            for topic_id in topic[1].split(','):
                topic_count[topic_id] += 1
        weights = np.array(range(self.n_classes))
        for i in range(self.n_classes):
            weights[i] = topic_count[self.index2topic[i]]
        return weights
    

if __name__ == '__main__':
    gvar = GlobalVariable()
    print gvar.char_vectors.syn0.shape
    print gvar.word_vectors.syn0.shape
    print len(gvar.topics)
    print len(gvar.question_eval)
    print len(gvar.question_train)
    print len(gvar.question_train_topic)
