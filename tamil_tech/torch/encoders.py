import torch
from torch import nn
import torch.nn.functional as F
from tamil_tech.torch.layers import *

class ConformerEncoder(nn.Module):
  def __init__(self, in_feat=128, conv_kernel_size=32, embed_dim=144, num_heads=4, dropout=0.1, num_encoders=16, n_feats=128, mask=None, batch_first=False):
    super(ConformerEncoder, self).__init__()

    self.conv_subsampling = ConvSubSampling(1, conv_kernel_size, transpose=batch_first)

    self.linear = nn.Linear(conv_kernel_size*in_feat, embed_dim)

    self.dropout = nn.Dropout(dropout)

    self.conformer_encoders = nn.Sequential(*[
      ConformerEncoderlayer(embed_dim, embed_dim, num_heads=num_heads, dropout=dropout, n_feats=n_feats) for _ in range(num_encoders)
    ])

    self.mask = mask

  def forward(self, x):
    x = self.conv_subsampling(x)
    x = F.gelu(x)
    x = self.linear(x)
    x = F.gelu(x)
    x = self.dropout(x)

    x = self.conformer_encoders(x)

    return x