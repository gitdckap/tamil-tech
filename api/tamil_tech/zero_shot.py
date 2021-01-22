import os
import warnings
import requests
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import torch
import shutil
import librosa
import torchvision
import torch.nn as nn
import soundfile as sf
import torch.nn.functional as F
from tamil_tech.torch.utils import *
from tamil_tech.torch.models import *

import numpy as np
import tensorflow as tf

from tensorflow_asr.utils import setup_environment, setup_devices

setup_environment()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})

setup_devices([0], cpu=False)

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer, SubwordFeaturizer
from tensorflow_asr.models.conformer import Conformer

class ConformerTamilASR(object):
    """
    Conformer S based ASR model
    """
    def __init__(self, path='ConformerS.h5'):
        # fetch and load the config of the model
        config = Config('tamil_tech/configs/conformer_new_config.yml', learning=True)

        # load speech and text featurizers
        speech_featurizer = TFSpeechFeaturizer(config.speech_config)
        text_featurizer = CharFeaturizer(config.decoder_config)

        # check if model already exists in given path, else download the model in the given path
        if os.path.exists(path):
          pass
        else:
          print("Downloading Model...")
          file_id = config.file_id
          download_file_from_google_drive(file_id, path)
          print("Downloaded Model Successfully...")
        
        # load model using config
        self.model = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
        # set shape of the featurizer and build the model
        self.model._build(speech_featurizer.shape)
        # load weights of the model
        self.model.load_weights(path, by_name=True)
        # display model summary
        self.model.summary(line_length=120)
        # set featurizers for the model
        self.model.add_featurizers(speech_featurizer, text_featurizer)

        print("Loaded Model...!")
    
    def read_raw_audio(self, audio, sample_rate=16000):
        # if audio path is given, load audio using librosa
        if isinstance(audio, str):
            wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
        
        # if audio file is in bytes, use soundfile to read audio
        elif isinstance(audio, bytes):
            wave, sr = sf.read(io.BytesIO(audio))
            
            # if audio is stereo, convert it to mono
            try:
                if wave.shape[1] >= 2:
                  wave = np.transpose(wave)[0][:]
            except:
              pass
            
            # get loaded audio as numpy array
            wave = np.asfortranarray(wave)

            # resampel to 16000 kHz
            if sr != sample_rate:
                wave = librosa.resample(wave, sr, sample_rate)
        
        # if numpy array, return audio
        elif isinstance(audio, np.ndarray):
            return audio
        
        else:
            raise ValueError("input audio must be either a path or bytes")
        return wave

    def bytes_to_string(self, array: np.ndarray, encoding: str = "utf-8"):
        # decode text array with utf-8 encoding
        return [transcript.decode(encoding) for transcript in array]

    def infer(self, path, greedy=True, return_text=False):
        # read the audio 
        signal = self.read_raw_audio(path)
        # expand dims to process for a single prediction
        signal = tf.expand_dims(self.model.speech_featurizer.tf_extract(signal), axis=0)
        # predict greedy
        if greedy:
          pred = self.model.recognize(features=signal)
        else:
          # preidct using beam search and language model
          pred = self.model.recognize_beam(features=signal, lm=True)

        if return_text:
          # return predicted transcription
          return self.bytes_to_string(pred.numpy())[0]
        
        # return predicted transcription
        print(self.bytes_to_string(pred.numpy())[0], end=' ')

class SubwordConformerTamilASR(object):
    """
    Subword Conformer S based ASR model
    """
    def __init__(self, path='ConformerS.h5'):
        # fetch and load the config of the model
        config = Config('tamil_tech/configs/subword_conformer_new.yml', learning=True)

        # load speech and text featurizers
        speech_featurizer = TFSpeechFeaturizer(config.speech_config)
        text_featurizer = SubwordFeaturizer.load_from_file(config.decoder_config, config.decoder_config['subwords'])

        # check if model already exists in given path, else download the model in the given path
        if os.path.exists(path):
          pass
        else:
          print("Downloading Model...")
          file_id = config.file_id
          download_file_from_google_drive(file_id, path)
          print("Downloaded Model Successfully...")
        
        # load model using config
        self.model = Conformer(**config.model_config, vocabulary_size=text_featurizer.num_classes)
        # set shape of the featurizer and build the model
        self.model._build(speech_featurizer.shape)
        # load weights of the model
        self.model.load_weights(path, by_name=True)
        # display model summary
        self.model.summary(line_length=120)
        # set featurizers for the model
        self.model.add_featurizers(speech_featurizer, text_featurizer)

        print("Loaded Model...!")
    
    def read_raw_audio(self, audio, sample_rate=16000):
        # if audio path is given, load audio using librosa
        if isinstance(audio, str):
            wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
        
        # if audio file is in bytes, use soundfile to read audio
        elif isinstance(audio, bytes):
            wave, sr = sf.read(io.BytesIO(audio))
            
            # if audio is stereo, convert it to mono
            try:
                if wave.shape[1] >= 2:
                  wave = np.transpose(wave)[0][:]
            except:
              pass
            
            # get loaded audio as numpy array
            wave = np.asfortranarray(wave)

            # resampel to 16000 kHz
            if sr != sample_rate:
                wave = librosa.resample(wave, sr, sample_rate)
        
        # if numpy array, return audio
        elif isinstance(audio, np.ndarray):
            return audio
        
        else:
            raise ValueError("input audio must be either a path or bytes")
        return wave

    def bytes_to_string(self, array: np.ndarray, encoding: str = "utf-8"):
        # decode text array with utf-8 encoding
        return [transcript.decode(encoding) for transcript in array]

    def infer(self, path, greedy=True, return_text=False):
        # read the audio 
        signal = self.read_raw_audio(path)
        # expand dims to process for a single prediction
        signal = tf.expand_dims(self.model.speech_featurizer.tf_extract(signal), axis=0)
        # predict greedy
        if greedy:
          pred = self.model.recognize(features=signal)
        else:
          # preidct using beam search and language model
          pred = self.model.recognize_beam(features=signal, lm=True)

        if return_text:
          # return predicted transcription
          return self.bytes_to_string(pred.numpy())[0]
        
        # return predicted transcription
        print(self.bytes_to_string(pred.numpy())[0], end=' ')
    
    def infer_dir(self, directory=None):
        for root, _, files in os.walk(directory):
            for file in files:
              print(f"Filename: {file} | ", end='')
              self.infer(os.path.join(root, file))
              print('')

class ConformerTamilASRLite(object):
    """
    Conformer S zero shot class for TFLite models
    """
    def __init__(self, greedy=False, path='conformer_s_v1.tflite'):
        self.greedy = greedy

        if os.path.exists(path):
          pass
        else:
          if self.greedy:
            print("Downloading Model...")
            file_id = '1GmBgu_q0nU_Q_borNsYs5b_1l2FXzVW3'
            download_file_from_google_drive(file_id, path)
            print("Downloaded Model Successfully...")
          else:
            print("Downloading Model...")
            file_id = '1GmBgu_q0nU_Q_borNsYs5b_1l2FXzVW3'
            download_file_from_google_drive(file_id, path)
            print("Downloaded Model Successfully...")
        
        self.model = tf.lite.Interpreter(model_path=path)

        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.blank = 0
        self.dim = 320

        print("Loaded Model...!")

    def read_raw_audio(self, audio, sample_rate=16000):
        if isinstance(audio, str):
            wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
        elif isinstance(audio, bytes):
            wave, sr = sf.read(io.BytesIO(audio))
            wave = np.asfortranarray(wave)
            if sr != sample_rate:
                wave = librosa.resample(wave, sr, sample_rate)
        elif isinstance(audio, np.ndarray):
            return audio
        else:
            raise ValueError("input audio must be either a path or bytes")
        return wave

    def infer(self, path):
        signal = self.read_raw_audio(path)
        self.model.resize_tensor_input(self.input_details[0]["index"], signal.shape)
        self.model.allocate_tensors()
        self.model.set_tensor(self.input_details[0]["index"], signal)
        self.model.set_tensor(
            self.input_details[1]["index"],
            tf.constant(self.blank, dtype=tf.int32)
        )
        self.model.set_tensor(
            self.input_details[2]["index"],
            tf.zeros([1, 2, 1, self.dim], dtype=tf.float32)
        )
        self.model.invoke()
        hyp = self.model.get_tensor(self.output_details[0]["index"])
        print("".join([chr(u) for u in hyp]), end=' ')
    
    def infer_dir(self, directory=None):
        for root, _, files in os.walk(directory):
            for file in files:
              print(f"Filename: {file} | ", end='')
              self.infer(os.path.join(root, file))
              print('')

class DeepTamilASR(object):
  """
  PyTorch - Custom Deep Speech 2 based ASR Model for Tamil
  """
  def __init__(self, use_gpu=True):
    # check GPU availability
    if use_gpu == True:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    else:
      self.device = torch.device("cpu")

    # load model on available device
    self.model = load_model().to(self.device)

    # set transforms for converting audio file to spectrogram
    self.transforms = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=128, normalized=True)
  
  def infer(self, filepath=None, verbose=1):
    # set model in evaluation mode
    self.model.eval()

    # laod the audio file
    audio, _ = torchaudio.load(filepath)

    # convert audio to mono if it is in stereo
    if audio.shape[0] == 2:
      audio = audio[0][:]

    # expand dims and take transpose
    audio = self.transforms(audio).squeeze(0).transpose(0, 1)

    audio = [audio]

    # pad audio tensor
    audio = nn.utils.rnn.pad_sequence(audio, batch_first=True).unsqueeze(1).transpose(2, 3)

    # predict for sequence
    output = self.model(audio.to(self.device))  
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)

    # decode sequence with vocabulary
    decoded_preds = GreedyDecoder(output.transpose(0, 1).type(torch.IntTensor), mode='infer')

    if verbose == 1:
      print(f"File: {filepath} | Transcription: {decoded_preds[0]}")
    
    else:
      return decoded_preds[0]
  
  def infer_dir(self, directory=None):
        for root, _, files in os.walk(directory):
            for file in files:
              print(f"Filename: {file} | ", end='')
              self.infer(os.path.join(root, file))
              print('')
 