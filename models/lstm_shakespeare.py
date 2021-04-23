import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from config import batch_size_train, device


class ModelLSTMShakespeare(nn.Module):
    def __init__(self):
        super(ModelLSTMShakespeare, self).__init__()
        self.embedding_len = 8
        self.seq_len = 80
        self.num_classes = 80
        self.n_hidden = 256
        self.batch_size = batch_size_train

        self.embeds = nn.Embedding(self.seq_len, self.embedding_len)
        self.multi_lstm = nn.LSTM(input_size=self.embedding_len, hidden_size=self.n_hidden, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, x, out_activation=False):
        x = x.to(torch.int64)
        x_ = self.embeds(x)
        h0 = torch.rand(2, x_.size(0), self.n_hidden).to(device)
        c0 = torch.rand(2, x_.size(0), self.n_hidden).to(device)
        activation, (h_n, c_n) = self.multi_lstm(x_,(h0,c0))

        fc_ = activation[:, -1, :]

        output = self.fc(fc_)
        if out_activation:
            return output, activation
        else:
            return output
