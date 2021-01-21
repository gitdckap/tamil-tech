import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from tamil_tech.torch.layers import *
from tamil_tech.torch.encoders import *
from tamil_tech.torch.decoders import *

class CustomCNNbiRNN(nn.Module):
    """
    ASR Model similar to Deep Speech
    """
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1, training=True):
        super(CustomCNNbiRNN, self).__init__()

        n_feats = n_feats//2

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
        x = self.in_cnn(x)
        x = self.residual_blocks(x)
        
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  
        
        x = x.transpose(1, 2) 
        x = self.pointwise(x)
        x = self.birnn_blocks(x)
        x = self.classifier(x)
        
        return x

class ExperimentalASR(nn.Module):
    """
    Diminishing Residual net based Bi RNN Model for ASR
    """
    def __init__(self, n_rnn_layers, rnn_dim, n_class, dropout=0.1, training=True, resnet='resnet18'):
        super(ExperimentalASR, self).__init__()

        if resnet == 'resnet18':
          self.resnet = resnet18(pretrained=False)
        elif resnet == 'resnet50':
          self.resnet = resnet50(pretrained=False)
        else:
          raise Exception("Invalid model passed. Choose resnet18 or resnet50")

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        n_inputs = self.resnet.fc.in_features

        for param in self.resnet.parameters():
              param.requires_grad = True

        _res = list(self.resnet.children())[:-2]
        self.resnet = nn.Sequential(*_res)

        self.pointwise = nn.Linear(2048, rnn_dim)
        self.birnn_blocks = nn.Sequential(*[BidirectionalRNN(rnn_dim=rnn_dim if i==0 else rnn_dim*2, hidden_size=rnn_dim, dropout=dropout, batch_first=i==0) for i in range(n_rnn_layers)])
        
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  
            Swish(),
            nn.Linear(rnn_dim, n_class)
        )
    
    def forward(self, x):      
        x = self.resnet(x)
                
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) 

        x = x.transpose(1, 2)
        x = self.pointwise(x)
        x = self.birnn_blocks(x)
        x = self.classifier(x)
        
        return x
        
class TamilASRModel(nn.Module):
    """
    ASR Model for Tamil
    """
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
            nn.Linear(rnn_dim*2, rnn_dim),  
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

class ConformerTransducer(nn.Module):
  def __init__(self, vocab_size=64, in_feat=128, conv_kernel_size=32, embed_dim=144, num_heads=4, dropout=0.1, n_feats=128, num_encoders=16, decoder_dim=320, num_decoders=1, decoder=nn.GRU, mask=None, batch_first=True):
    super(ConformerTransducer, self).__init__()

    self.encoder = ConformerEncoder(in_feat, conv_kernel_size, embed_dim, num_heads, dropout, num_encoders, n_feats, mask, batch_first)

    self.decoder = decoder(embed_dim, decoder_dim, num_layers=num_decoders, dropout=dropout, bidirectional=False, batch_first=batch_first)

    self.fc = nn.Sequential(nn.Linear(decoder_dim, decoder_dim),
                            Swish(),
                            nn.Linear(decoder_dim, vocab_size))
  
  def forward(self, x):
    x = self.encoder(x)
    x, states = self.decoder(x)
    x = self.fc(x)

    return x, states

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        return pred, gold

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model, LFR_m, LFR_n = cls.load_model_from_package(package)
        return model, LFR_m, LFR_n

    @classmethod
    def load_model_from_package(cls, package):
        encoder = Encoder(package['d_input'],
                          package['n_layers_enc'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          pe_maxlen=package['pe_maxlen'])
        decoder = Decoder(package['sos_id'],
                          package['eos_id'],
                          package['vocab_size'],
                          package['d_word_vec'],
                          package['n_layers_dec'],
                          package['n_head'],
                          package['d_k'],
                          package['d_v'],
                          package['d_model'],
                          package['d_inner'],
                          dropout=package['dropout'],
                          tgt_emb_prj_weight_sharing=package['tgt_emb_prj_weight_sharing'],
                          pe_maxlen=package['pe_maxlen'],
                          )
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        LFR_m, LFR_n = package['LFR_m'], package['LFR_n']
        return model, LFR_m, LFR_n

    @staticmethod
    def serialize(model, optimizer, epoch, LFR_m, LFR_n, tr_loss=None, cv_loss=None):
        package = {
            # Low Frame Rate Feature
            'LFR_m': LFR_m,
            'LFR_n': LFR_n,
            # encoder
            'd_input': model.encoder.d_input,
            'n_layers_enc': model.encoder.n_layers,
            'n_head': model.encoder.n_head,
            'd_k': model.encoder.d_k,
            'd_v': model.encoder.d_v,
            'd_model': model.encoder.d_model,
            'd_inner': model.encoder.d_inner,
            'dropout': model.encoder.dropout_rate,
            'pe_maxlen': model.encoder.pe_maxlen,
            # decoder
            'sos_id': model.decoder.sos_id,
            'eos_id': model.decoder.eos_id,
            'vocab_size': model.decoder.n_tgt_vocab,
            'd_word_vec': model.decoder.d_word_vec,
            'n_layers_dec': model.decoder.n_layers,
            'tgt_emb_prj_weight_sharing': model.decoder.tgt_emb_prj_weight_sharing,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package