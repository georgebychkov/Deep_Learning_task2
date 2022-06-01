import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import *

class LSTM(nn.Module):
    
    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                               drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(n_hidden, len(self.chars))
        
        self.init_weights()
      
    
    def forward(self, x, hc):
        
        x, (h, c) = self.lstm(x, hc)
        
        x = self.dropout(x)
        
        x = x.reshape(x.size()[0]*x.size()[1], self.n_hidden)
        
        x = self.fc(x)
        
        return x, (h, c)
    
    
    def predict(self, char, h=None, device="cpu", top_k=None):
        self.to(device)
        
        if h is None:
          h = self.init_hidden(1)
        
        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))
        
        inputs = torch.from_numpy(x)
        
        inputs = inputs.to(device)
        
        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        
        p = p.cpu()
        
        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()

        if top_k == 1:
            top_ch = top_ch.reshape((1,))
            char = top_ch[0]
        else:
            char = np.random.choice(top_ch, p=p/p.sum())
            
        return self.int2char[char], h
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())
        
        