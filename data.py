from torch.utils.data import Dataset
from zutil.config import Config
from configure import GlobalVariable
from torch.autograd import Variable
import numpy as np
import random
import torch

class QuestionDataset(Dataset):
    def __init__(self, config, gvar):
        self.gvar = gvar
        self.config = config
        
        if self.config.mode == 'inference':
            self.questions = self.gvar.question_eval
        elif self.config.mode in {'train', 'val'}:
            self.questions = self.gvar.question_train
        
    def __len__(self):
        return len(self.questions)

    def __get_question__(self, index):
        if self.config.question_type == 'title':
            question_char = self.questions[index][1]
            question_word = self.questions[index][2]
        elif self.config.question_type == 'description':
            question_char = self.questions[index][3]
            question_word = self.questions[index][4]
        question_char = np.array([self.gvar.char_vectors[key] for key in question_char.split(',') 
            if self.gvar.char_vectors.vocab.has_key(key)])
        question_word = np.array([self.gvar.word_vectors[key] for key in question_word.split(',') 
            if self.gvar.word_vectors.vocab.has_key(key)])

        return question_char, question_word

    def __getitem__(self, index):
        question_char, question_word = self.__get_question__(index)
        if self.config.model_type == 'baseline':
            if question_char.shape[0] != 0 and question_word.shape[0] != 0:
                question_char = question_char.mean(0)
                question_word = question_word.mean(0)
        if self.config.mode == 'inference':
            return question_char, question_word, index
        topic_indices = [self.gvar.topic2index[w] for w in \
                self.gvar.question_train_topic[index][1].split(',')]
        target = np.zeros(self.config.n_classes)
        target[topic_indices] = 1
        return question_char, question_word, target.astype('int')

    def __iter__(self):
        if self.config.mode == 'inference':
            for index in range(self.__len__()):
                yield self.__getitem__(index)
        else:
            cut_point = int(self.__len__() * self.config.split_rate)
            if self.config.mode == 'train':
                start, end = 0, cut_point - 1
            elif self.config.mode == 'val':
                start, end = cut_point, self.__len__() - 1
            while True:
                # randomly choose one item every time
                yield self.__getitem__(random.randint(start, end))

    def next_batch(self):
        batch_data = []
        for t, data in enumerate(self):
            if data[0].shape[0] == 0 or data[1].shape[0] == 0:
                continue  # if no such word/char in dictionary of word2vec
            batch_data.append(data)
            if len(batch_data) == self.config.batch_size:
                yield self.split_batch(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield self.split_batch(batch_data)
                
    def split_batch(self, batch_data):
        questions_char = [question for question, _, _ in batch_data]
        questions_word = [question for _, question, _ in batch_data]
        targets = [target for _, _, target in batch_data] # target or indices ?
        targets = Variable(torch.from_numpy(np.array(targets))).float()
        if self.config.gpu:
            targets = targets.cuda()
        
        # if baseline
        if self.config.model_type == 'baseline':
            questions_char = Variable(torch.from_numpy(np.array(questions_char))).float()
            questions_word = Variable(torch.from_numpy(np.array(questions_word))).float()
            questions = torch.cat([questions_char, questions_word], 1)
            if self.config.gpu:
                questions = questions.cuda()
            return questions, targets

        # else if LSTM/RNN model, need padding
        # need to fix
        questions = questions_char
        lengths = np.array([min(self.config.max_qlen, question.shape[0]) for question in questions])
        index = lengths.argsort()[::-1]     # re-sort by lengths
        lengths, questions, targets = lengths[index], questions[index], targets[index]
        padded_questions = np.zeros((self.config.batch_size, self.config.max_qlen, \
            self.gvar.word_vectors.syn0.shape[1]))
        for i in range(self.config.batch_size):
            padded_questions[i, 0:lengths[i], :] = questions[i][0:lengths[i]]
        padded_questions = Variable(torch.from_numpy(padded_questions)).float()

        if self.config.gpu:
            padded_questions = padded_questions.cuda()

        return (padded_questions, lengths), targets
            
    
if __name__ == '__main__':
    gvar = GlobalVariable()
    config = Config('parameter.json', model_type='baseline')
    dataset = QuestionDataset(config, gvar)
    for questions, targets in dataset.next_batch():
        print questions
        print targets
        break

    print '---------------------'
    config = Config('parameter.json', model_type='lstm')
    dataset = QuestionDataset(config, gvar)
    for (questions, targets), lengths in dataset.next_batch():
        print questions
        print targets
        print lengths
        break
