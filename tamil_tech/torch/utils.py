import warnings
warnings.filterwarnings("ignore")

import os
import time
import math
import json
import wave
import torch
import requests
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
from tamil import utf8
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from tamil_tech.torch.models import *
from tamil_tech.torch.transforms import *
from torch.utils.data import DataLoader, Dataset

def get_words(sentence):
  return sentence.split()

def calculate_distance(word1, word2):
  return len(word1.replace(word2, ''))

def correct_spelling(predictions):
  words = get_words(predictions)
  correct_unigrams = []
  counts = np.load('lm_probabilities.npy', allow_pickle=True)
  counts = dict(counts.item())

  for word in words:      
      if counts.get(word, 0):
          correct_unigrams.append(word)
      else:
        # The word did not occur in the training data!
          # Now find a good replacement
          minimum_distance = math.inf
          best_replacement = None
          for vocab_word in counts:
              distance = calculate_distance(vocab_word, word)
              # Make sure the vocabulary word is not the same as the spelling error and make sure it is better than the
              # result found so far
              if 0 < distance < minimum_distance:
                  minimum_distance = distance
                  best_replacement = vocab_word
          # Show the correction
          correct_unigrams.append(best_replacement)
  
  return ''.join(utf8.get_letters(' '.join(correct_unigrams)))

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)

def _levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]

def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    if ignore_case == True:
        reference = reference
        hypothesis = hypothesis

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)

def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    if ignore_case == True:
        reference = reference
        hypothesis = hypothesis

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)

def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer

def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

class TextTransform:
    def __init__(self):
        with open('tamil_tech/vocabularies/tamil_tokenized_v3.json', 'r') as fp:
          tokenizer = json.load(fp)
        
        self.char_map = tokenizer['word_index']
        self.index_map = tokenizer['index_word']

    def text_to_int(self, text):
        int_sequence = []
        text = list(text)
        
        for c in text:
          try:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            
            int_sequence.append(ch)
          
          except Exception as e:
            pass
        
        return int_sequence

    def int_to_text(self, labels):
        string = []
        
        for i in labels:
            string.append(self.index_map[str(int(i))])
        
        return ''.join(utf8.get_letters(''.join(string).replace('<SPACE>', ' ')))

text_transform = TextTransform()

class OpenSLRDataset(Dataset):
  def __init__(self, directory, filename, mode='train'):
    if mode == 'train':
      self.transforms = train_audio_transforms
    elif mode in ['test', 'dev']:
      self.transforms = valid_audio_transforms
    else:
      raise Exception("Invalid mode selected. Select train, dev, or test")

    self.directory = directory
    self.filename = filename

    self.df = pd.read_csv(os.path.join(directory, filename))

    self.filenames = self.df.filename.values.tolist()
    self.category = self.df.category.values.tolist()
    self.speakers = self.df.speaker.values.tolist()
    self.labels = self.df.texts.values.tolist()
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    audio_path = self.directory + '/' + self.category[index] + '/' + self.filenames[index] + '.wav'
    utterances = self.labels[index]

    if self.speakers[index] == 'Male':
      speaker = 0
    else:
      speaker = 1

    wave, sr = torchaudio.load(audio_path)
    
    return wave, sr, utterances, speaker, index, index

class TamilAudioDataset(Dataset):
  def __init__(self, directory, filename, mode='train'):
    if mode == 'train':
      self.transforms = train_audio_transforms
    elif mode in ['test', 'dev']:
      self.transforms = valid_audio_transforms
    else:
      raise Exception("Invalid mode selected. Select train, dev, or test")

    self.directory = directory
    self.filename = filename

    self.df = pd.read_csv(os.path.join(directory, filename), delimiter='\t', encoding='utf-8')

    self.filenames = self.df.PATH.values.tolist()
    self.labels = self.df.TRANSCRIPT.values.tolist()
    self.durations = self.df.DURATION.values.tolist()
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, index):
    audio_path = self.directory + '/' + self.filenames[index]
    utterances = self.labels[index]
    duration = self.durations[index]

    wave, sr = torchaudio.load(audio_path)
    
    return wave, sr, utterances, duration, index, index

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return torch.Tensor(spectrograms), torch.Tensor(labels), torch.IntTensor(input_lengths), torch.IntTensor(label_lengths)

def GreedyDecoder(output, labels=None, label_lengths=None, blank_label=0, collapse_repeated=True, mode='valid'):
  if mode == 'valid':
    arg_maxes = torch.argmax(output, dim=2)
    
    decodes = []
    targets = []
    
    for i, args in enumerate(arg_maxes):
      decode = []
    
      targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
    
      for j, index in enumerate(args):
        if index != blank_label:
          if collapse_repeated and j != 0 and index == args[j -1]:
            continue
    
          decode.append(index.item())
    
      decodes.append(text_transform.int_to_text(decode))
    
    return decodes, targets
  
  elif mode == 'infer':
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []

    for i, args in enumerate(arg_maxes):
      decode = []
    
      for j, index in enumerate(args):
        if index != blank_label:
          if collapse_repeated and j != 0 and index == args[j -1]:
            continue
          
          decode.append(index.item())

      decodes.append(text_transform.int_to_text(decode))
    
    return decodes

def save_checkpoint(state, filename='./models/tamil_tts.pt'):
  torch.save(state, filename)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_model(model_path='asr_model.pt'):
  hparams = {
          "n_cnn_layers": 3,
          "n_rnn_layers": 5,
          "rnn_dim": 512,
          "n_class": 64,
          "n_feats": 128,
          "stride": 2,
          "dropout": 0.1
  }

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  model = TamilASRModel(
    hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
    hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
  ).to(device)

  if os.path.exists(model_path):
    pass
  
  else:
    print("Downloading Model...")
    file_id = '1-qqd0bp-Vh_xJlDq9OccaJbDP8rEotRo'
    download_file_from_google_drive(file_id, model_path)
    print("Downloaded Model Successfully...")

  model.load_state_dict(torch.load(model_path, map_location=device))

  print("Loaded Model!")

  return model

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
    # N x Ti, lt(1) like not operation
    pad_mask = non_pad_mask.squeeze(-1).lt(1)
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1)
    return attn_mask

IGNORE_ID = -1


def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


def process_dict(dict_path):
    with open(dict_path, 'rb') as f:
        dictionary = f.readlines()
    char_list = [entry.decode('utf-8').split(' ')[0]
                 for entry in dictionary]
    sos_id = char_list.index('<sos>')
    eos_id = char_list.index('<eos>')
    return char_list, sos_id, eos_id

def parse_hypothesis(hyp, char_list):
    """Function to parse hypothesis
    :param list hyp: recognition hypothesis
    :param list char_list: list of characters
    :return: recognition text strinig
    :return: recognition token strinig
    :return: recognition tokenid string
    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp['yseq'][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp['score'])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace('<space>', ' ')

    return text, token, tokenid, score


def add_results_to_json(js, nbest_hyps, char_list):
    """Function to add N-best results to json
    :param dict js: groundtruth utterance dict
    :param list nbest_hyps: list of hypothesis
    :param list char_list: list of characters
    :return: N-best results added utterance dict
    """
    # copy old json info
    new_js = dict()
    new_js['utt2spk'] = js['utt2spk']
    new_js['output'] = []

    for n, hyp in enumerate(nbest_hyps, 1):
        # parse hypothesis
        rec_text, rec_token, rec_tokenid, score = parse_hypothesis(
            hyp, char_list)

        # copy ground-truth
        out_dic = dict(js['output'][0].items())

        # update name
        out_dic['name'] += '[%d]' % n

        # add recognition results
        out_dic['rec_text'] = rec_text
        out_dic['rec_token'] = rec_token
        out_dic['rec_tokenid'] = rec_tokenid
        out_dic['score'] = score

        # add to list of N-best result dicts
        new_js['output'].append(out_dic)

        # show 1-best result
        if n == 1:
            print('groundtruth: %s' % out_dic['text'])
            print('prediction : %s' % out_dic['rec_text'])

    return new_js