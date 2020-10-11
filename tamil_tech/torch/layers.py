import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import TimeMasking, FrequencyMasking

class CustomSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish(nn.Module):
    def forward(self, x):
        return CustomSwish.apply(x)

class SpecAug(nn.Module):
    def __init__(self, freq_len=1, time_len=10, freq_mask=27, time_mask=100, training=True):
        super(SpecAug, self).__init__()

        self.training = training

        self.freq_len = freq_len
        self.time_len = time_len

        self.freq_mask = freq_mask
        self.time_mask = time_mask

        self.freq_aug = FrequencyMasking(freq_mask_param=self.freq_len)
        self.time_aug = TimeMasking(time_mask_param=self.time_len)
    
    def forward(self, x):
        if self.training:
            x = self.freq_aug(x, mask_value=self.freq_mask)
            x = self.time_aug(x, mask_value=self.time_mask)

        return x

class LayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous() 
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, n_feats, dropout=0.1):
        super(ResidualBlock, self).__init__()

        self.ln1 = LayerNorm(n_feats)
        self.dropout1 = nn.Dropout(p=dropout)
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.swish1 = Swish()
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.ln2 = LayerNorm(n_feats)
        self.dropout2 = nn.Dropout(p=dropout)   
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)    
        self.swish2 = Swish()     
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x  
        
        x = self.ln1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.swish1(x)
        x = self.bn1(x)
        
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x = self.swish2(x)
        x = self.bn2(x)
        
        x += residual
        
        return x 

class BidirectionalRNN(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first, rnn=nn.GRU):
        super(BidirectionalRNN, self).__init__()

        self.BiGRU = rnn(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        
        self.layer_norm = nn.LayerNorm(rnn_dim)

        self.swish = Swish()
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.swish(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        x = F.gelu(x)

        return x

class LayerNormalization(nn.Module):
    def __init__(self, n_feats):
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() 

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(Residual, self).__init__()

        self.layer_norm1 = LayerNormalization(n_feats)
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.layer_norm2 = LayerNormalization(n_feats)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout2 = nn.Dropout(dropout)      
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x 
        
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = F.gelu(self.bn1(x))
        
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x = F.gelu(self.bn2(x))
        
        x += residual
        
        return x 

class BiGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BiGRU, self).__init__()

        self.layer_norm = nn.LayerNorm(rnn_dim)

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        
        return x