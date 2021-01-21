import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchaudio.transforms import TimeMasking, FrequencyMasking

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomSwish(torch.autograd.Function):
    """
    Swish activation class with backward function
    """
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
    """
    Swish activation
    """
    def forward(self, x):
        return CustomSwish.apply(x)

class SpecAug(nn.Module):
    """
    Spectrogram Augmentation class
    """
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
    """
    Layer Normalization class
    """
    def __init__(self, n_feats):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous() 
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() 

class ResidualBlock(nn.Module):
    """
    Residual Network based feature extraction class for Audio Spectrogram
    """
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
    """
    Bi-directional RNN class
    """
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
    """
    Layer Normalization
    """
    def __init__(self, n_feats):
        super(LayerNormalization, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() 

class Residual(nn.Module):
    """
    Residual feature extraction
    """
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
    """
    Bi directional GRU class
    """
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

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad + (kernel_size + 1) % 2)

# Following are available in PyTorch by default

class Transpose(nn.Module):
    """
    Transpose as a Layer
    """
    def __init__(self, dims):
        super(Transpose, self).__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    """
    GLU activation
    """
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    """
    Depth Wise 1-D Convolution
    """
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super(DepthWiseConv1d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)
        self.conv_out = nn.Conv1d(chan_out, chan_out, 1)

    def forward(self, x):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        return self.conv_out(x)

class ConformerConvModule(nn.Module):
    """
    Conformer transducer - Conv Module
    """
    def __init__(self, dim, causal = False, expansion_factor = 2, kernel_size = 31, dropout = 0.):
        super(ConformerConvModule, self).__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        residual = x
        x = self.net(x)
        return x + residual

class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        
        scores = query.matmul(key.transpose(-2, -1)) / torch.sqrt(dk)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        
        return attention.matmul(value)

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, in_features, head_num, bias=True, activation=F.relu):
    super(MultiHeadSelfAttention, self).__init__()
    
    if in_features % head_num != 0:
        raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
    
    self.in_features = in_features
    self.head_num = head_num
    self.activation = activation
    self.bias = bias
    self.linear_q = nn.Linear(in_features, in_features, bias)
    self.linear_k = nn.Linear(in_features, in_features, bias)
    self.linear_v = nn.Linear(in_features, in_features, bias)
    self.linear_o = nn.Linear(in_features, in_features, bias)

  def forward(self, q, k, v, mask=None):
      q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
      if self.activation is not None:
          q = self.activation(q)
          k = self.activation(k)
          v = self.activation(v)

      q = self._reshape_to_batches(q)
      k = self._reshape_to_batches(k)
      v = self._reshape_to_batches(v)
      if mask is not None:
          mask = mask.repeat(self.head_num, 1, 1)
      y = ScaledDotProductAttention()(q, k, v, mask)
      y = self._reshape_from_batches(y)

      y = self.linear_o(y)
      if self.activation is not None:
          y = self.activation(y)
      return y

  @staticmethod
  def gen_history_mask(x):
      batch_size, seq_len, _ = x.size()
      return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

  def _reshape_to_batches(self, x):
      batch_size, seq_len, in_feature = x.size()
      sub_dim = in_feature // self.head_num
      return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
              .permute(0, 2, 1, 3)\
              .reshape(batch_size * self.head_num, seq_len, sub_dim)

  def _reshape_from_batches(self, x):
      batch_size, seq_len, in_feature = x.size()
      batch_size //= self.head_num
      out_dim = in_feature * self.head_num
      return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
              .permute(0, 2, 1, 3)\
              .reshape(batch_size, seq_len, out_dim)

  def extra_repr(self):
      return 'in_features={}, head_num={}, bias={}, activation={}'.format(
          self.in_features, self.head_num, self.bias, self.activation,
      )

class FeedForward(nn.Module):
  def __init__(self, in_feat, out_feat, dropout, n_feats):
    super(FeedForward, self).__init__()

    self.layer_norm = nn.LayerNorm(n_feats)
    self.linear1 = nn.Linear(in_feat, out_feat)
    self.swish = Swish()
    self.dropout1 = nn.Dropout(dropout)
    self.linear2 = nn.Linear(out_feat, out_feat)
    self.dropout2 = nn.Dropout(dropout)
  
  def forward(self, x):
    residual = x

    x = x.transpose(1, 2).contiguous()
    x = self.layer_norm(x)
    x = x.transpose(1, 2).contiguous()
    x = self.linear1(x)
    x = self.swish(x)
    x = self.dropout1(x)
    x = self.linear2(x)
    x = self.dropout2(x)

    return residual + x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class RelativePositionalEmbedding(nn.Module):
    def __init__(self, d):
        super(RelativePositionalEmbedding, self).__init__()
        self.d = d
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d, 2.0) / d))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):  # input size (seq, )
        # outer product
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]

class MultiheadedAttention(nn.Module):
    """
    Narrow multiheaded attention. Each attention head inspects a 
    fraction of the embedding space and expresses attention vectors for each sequence position as a weighted average of all (earlier) positions.
    """
    def __init__(self, d_model, heads=8, dropout=0.1, relative_pos=True):

        super().__init__()
        if d_model % heads != 0:
            raise Exception("Number of heads does not divide model dimension")
        
        self.d_model = d_model
        self.heads = heads
        s = d_model // heads
        self.linears = torch.nn.ModuleList([nn.Linear(s, s, bias=False) for i in range(3)])
        self.recombine_heads = nn.Linear(heads * s, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.max_length = 1024
        #relative positional embeddings
        self.relative_pos = relative_pos
        if relative_pos:
            self.Er = torch.randn([heads, self.max_length, s])
        else:
            self.Er = None

    def forward(self, x, mask):
        #batch size, sequence length, embedding dimension
        b, t, e = x.size()
        h = self.heads
        #each head inspects a fraction of the embedded space
        #head dimension
        s = e // h
        #start index of position embedding
        embedding_start = self.max_length - t
        x = x.view(b,t,h,s)
        queries, keys, values = [w(x).transpose(1,2)
                for w, x in zip(self.linears, (x,x,x))]
        if self.relative_pos:
            #apply same position embeddings across the batch
            #Is it possible to apply positional self-attention over
            #only half of all relative distances?
            Er  = self.Er[:, embedding_start:, :].unsqueeze(0)
            QEr = torch.matmul(queries, Er.transpose(-1,-2))
            QEr = self._mask_positions(QEr)
            #Get relative position attention scores
            #combine batch with head dimension
            SRel = self._skew(QEr).contiguous().view(b*h, t, t)
        else:
            SRel = torch.zeros([b*h, t, t])
        queries, keys, values = map(lambda x: x.contiguous()\
                .view(b*h, t, s), (queries, keys, values))
        #Compute scaled dot-product self-attention
        #scale pre-matrix multiplication   
        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))

        scores = torch.bmm(queries, keys.transpose(1, 2))
        scores = scores + SRel
        #(b*h, t, t)

        subsequent_mask = torch.triu(torch.ones(1, t, t),
                1)
        scores = scores.masked_fill(subsequent_mask == 1, -1e9)
        if mask is not None:
            mask = mask.repeat_interleave(h, 0)
            wtf = (mask == 0).nonzero().transpose(0,1)
            scores[wtf[0], wtf[1], :] = -1e9
        
        #Convert scores to probabilities
        attn_probs = F.softmax(scores, dim=2)
        attn_probs = self.dropout(attn_probs)
        #use attention to get a weighted average of values
        out = torch.bmm(attn_probs, values).view(b, h, t, s)
        #transpose and recombine attention heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)
        #last linear layer of weights
        return self.recombine_heads(out)

    def _mask_positions(self, qe):
        #QEr is a matrix of queries (absolute position) dot distance embeddings (relative pos).
        #Mask out invalid relative positions: e.g. if sequence length is L, the query at
        #L-1 can only attend to distance r = 0 (no looking backward).
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]

class ConformerMHSA(nn.Module):
  def __init__(self, embed_dim, num_heads, dropout=0.1):
    super(ConformerMHSA, self).__init__()

    self.ln = nn.LayerNorm(embed_dim)
    self.self_attention = MultiheadedAttention(embed_dim, heads=num_heads, dropout=dropout)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x, mask=None):
    residual = x

    x = self.ln(x)
    x = self.self_attention(x, mask)

    return x + residual

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class ConvSubSampling(nn.Module):
  def __init__(self, in_feat, out_feat, kernel_size=(3, 3), transpose=False):
    super(ConvSubSampling, self).__init__()

    self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(out_feat)

    self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=kernel_size, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_feat)

    self.transpose = transpose
  
  def forward(self, x):
    x = self.conv1(x)
    x = F.gelu(x)
    x = self.bn1(x)

    x = self.conv2(x)
    x = F.gelu(x)
    x = self.bn2(x)

    sizes = x.size()
    x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
    
    if self.transpose:
      x = x.transpose(1, 2) # (batch, time, feature)

    return x

class ConformerEncoderlayer(nn.Module):
  def __init__(self, in_feat, out_feat, num_heads, dropout, n_feats):
    super(ConformerEncoderlayer, self).__init__()

    self.feed_forward1 = FeedForward(in_feat, out_feat, dropout, n_feats)

    self.mhsa = ConformerMHSA(out_feat, num_heads, dropout)

    self.conv = ConformerConvModule(out_feat)

    self.feed_forward2 = FeedForward(out_feat, out_feat, dropout, n_feats)

    self.ln = nn.LayerNorm(out_feat)

  def forward(self, x):
    x = self.feed_forward1(x)
    x = self.mhsa(x)
    x = self.conv(x)
    x = self.feed_forward2(x)
    x = self.ln(x)

    return x

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),
                                                   attn_dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class PositionalEncoding(nn.Module):
    """Implement the positional encoding (PE) function.
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        """
        Args:
            input: N x T x D
        """
        length = input.size(1)
        return self.pe[:, :length]


class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


# Another implementation
class PositionwiseFeedForwardUseConv(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForwardUseConv, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output