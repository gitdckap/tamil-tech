import torch
import torch.nn as nn
import torch.nn.functional as F
from tamil_tech.layers.torch import *

class CustomCNNbiRNN(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, training=True):
        super(CustomCNNbiRNN, self).__init__()

        n_feats = n_feats//2

        self.spec_aug = SpecAug(training=training)

        self.in_cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(32, 32, kernel=3, stride=1, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])

        self.pointwise = nn.Linear(n_feats*32, rnn_dim)

        self.birnn_blocks = nn.Sequential(*[BidirectionalRNN(rnn_dim=rnn_dim if i==0 else rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, batch_first=i==0) for i in range(n_rnn_layers)])
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  
            Swish(),
            nn.Linear(rnn_dim, n_class)
        )
    
    def forward(self, x):
        x = self.spec_aug(x)
        
        x = self.in_cnn(x)
        x = self.residual_blocks(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  
        
        x = x.transpose(1, 2) 
        x = self.pointwise(x)
        x = self.birnn_blocks(x)
        x = self.classifier(x)
        
        return x

class TamilASRModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, training=True):
        super(TamilASRModel, self).__init__()
        
        n_feats = n_feats//2

        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        self.rescnn_layers = nn.Sequential(*[
            Residual(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats) 
            for _ in range(n_cnn_layers)
        ])

        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        
        self.birnn_layers = nn.Sequential(*[
            BiGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) 
        x = x.transpose(1, 2) 
        
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        
        return x