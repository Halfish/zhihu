from zutil.config import Config
import torch.nn as nn
import torch
import random

class TextModelModule(nn.Module):
    '''
    handle text with various model, like w2v baseline, lstm, cnn
    '''
    def __init__(self, config):
        super(TextModelModule, self).__init__()
        self.config = config
        if self.config.model_type in {'lstm', 'rnn'}:
            self.lstm = nn.LSTM(self.config.emb_dim, self.config.rnn_hidden_size, \
                    self.config.rnn_hidden_layers, batch_first=True)
            flatten_size = self.config.rnn_hidden_size
        elif self.config.model_type == 'baseline':
            flatten_size = self.config.emb_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(flatten_size, self.config.mlp_hidden_size),
            nn.BatchNorm1d(self.config.mlp_hidden_size),
            nn.ReLU(),
            #nn.Dropout(self.config.mlp_dropout),
            nn.Linear(self.config.mlp_hidden_size, self.config.mlp_hidden_size),
            nn.BatchNorm1d(self.config.mlp_hidden_size),
            nn.ReLU(),

            nn.Linear(self.config.mlp_hidden_size, self.config.mlp_hidden_size),
            nn.BatchNorm1d(self.config.mlp_hidden_size),
            nn.ReLU(),

            nn.Linear(self.config.mlp_hidden_size, self.config.n_classes)
        )
        self.training = True
        
    def forward(self, inputs):
        if self.config.model_type in {'lstm', 'rnn'}:
            questions, lengths = inputs 
            outputs, _ = self.lstm(questions)
            outputs = torch.cat([outputs[i][lengths[i]-1].unsqueeze(0) for i in range(len(lengths))], 0)
        elif self.config.model_type == 'baseline':
            outputs = inputs
        outputs = self.mlp(outputs)
        return outputs
        
    def train(self):
        if self.config.model_type in {'lstm', 'rnn'}:
            self.lstm.train()
        self.mlp.train()
        self.training = True
        
    def eval(self):
        if self.config.model_type in {'lstm', 'rnn'}:
            self.lstm.eval()
        self.mlp.eval()
        self.training = False


if __name__ == '__main__':
    # test for lstm model
    config = Config('parameter.json', model_type='lstm')
    model = TextModelModule(config)
    print model
    questions = torch.autograd.Variable(torch.randn(config.batch_size, config.max_qlen, config.emb_dim))
    lengths = [random.randint(0, config.max_qlen - 1) for i in range(config.batch_size)]
    print questions.data.size()
    print lengths

    inputs = (questions, lengths)
    outputs = model(inputs)
    print outputs.data.size()
    print outputs.data[0:10, 0:10]

    # test for baseline
    print '-----------------------'
    config = Config('parameter.json', model_type='baseline')
    model = TextModelModule(config)
    print model
    questions = torch.autograd.Variable(torch.randn(config.batch_size, config.max_qlen, config.emb_dim))
    outputs = model(questions)
    print outputs.data.size()
    print outputs.data[0:10, 0:10]
