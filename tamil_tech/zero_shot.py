import os
import warnings

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
from tamil_tech.tf.utils import TFSpeechFeaturizer, CharFeaturizer, preprocess_paths, check_key_in_dict, append_default_keys_dict, CONFORMER_L, CONFORMER_M, CONFORMER_S, CONFORMER_S_V2, CONFORMER_S_UPDATED, CONFORMER_S_UPDATED_V1, CONFORMER_S_UPDATED_V11, load_conformer_model, download_file_from_google_drive
from tamil_tech.tf.models import Conformer

class ConformerTamilASRLite(object):
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

class ConformerTamilASR(object):
    def __init__(self, greedy=False, path='Conformer_S_updated_v1_1.h5'):
        config = CONFORMER_S_UPDATED_V11

        speech_featurizer = TFSpeechFeaturizer(config["speech_config"])
        text_featurizer = CharFeaturizer(config["decoder_config"])

        if os.path.exists(path):
          pass
        else:
          print("Downloading Model...")
          file_id = config["file_id"]
          download_file_from_google_drive(file_id, path)
          print("Downloaded Model Successfully...")
        
        self.model = Conformer(
            vocabulary_size=text_featurizer.num_classes,
            **config["model_config"]
        )
        self.model._build(speech_featurizer.shape)
        self.model.load_weights(path, by_name=True)
        self.model.add_featurizers(speech_featurizer, text_featurizer)

        self.greedy = greedy

        print("Loaded Model...!")
    
    def read_raw_audio(self, audio, sample_rate=16000):
        if isinstance(audio, str):
            wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate)
        
        elif isinstance(audio, bytes):
            wave, sr = sf.read(io.BytesIO(audio))
            
            try:
                if wave.shape[1] >= 2:
                  wave = np.transpose(wave)[0][:]
            except:
              pass
            
            wave = np.asfortranarray(wave)

            if sr != sample_rate:
                wave = librosa.resample(wave, sr, sample_rate)
        
        elif isinstance(audio, np.ndarray):
            return audio
        
        else:
            raise ValueError("input audio must be either a path or bytes")
        return wave

    def bytes_to_string(self, array: np.ndarray, encoding: str = "utf-8"):
        return [transcript.decode(encoding) for transcript in array]

    def infer(self, path, return_text=False):
        signal = self.read_raw_audio(path)
        signal = tf.expand_dims(self.model.speech_featurizer.tf_extract(signal), axis=0)
        if self.greedy:
          pred = self.model.recognize(features=signal)
        else:
          pred = self.model.recognize_beam(features=signal, lm=True)

        if return_text:
          return self.bytes_to_string(pred.numpy())[0]
        
        print(self.bytes_to_string(pred.numpy())[0], end=' ')
    
    def infer_dir(self, directory=None):
      for root, _, files in os.walk(directory):
          for file in files:
            print(f"Filename: {file} | ", end='')
            self.infer(os.path.join(root, file))
            print('')
    
    def record_and_stream(self):
        print("Creating Audio stream...\nSpeak!")

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024)
            
        while True:
            try:
                frames = []

                for _ in range(0, int(16000 / 1024 * 7)):
                    data = self.stream.read(1024)
                    frames.append(data)
                        
                wf = wave.open(f'tamil_tech/__pycache__/0.wav', 'wb')
                wf.setnchannels(1)
                wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(frames))
                wf.close()

                self.infer(f'tamil_tech/__pycache__/0.wav')

            except Exception as e:
                print(e)
                break

            except KeyboardInterrupt:
                self.stream.stop_stream()
                self.stream.close()
                self.p.terminate()
                break
        
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    
    def __call__(self):
        self.record_and_stream()

class DeepTamilASR(object):
  def __init__(self, use_gpu=True):
    if use_gpu == True:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    else:
      self.device = torch.device("cpu")

    self.model = load_model().to(self.device)

    self.p = pyaudio.PyAudio()

    self.transforms = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=128, normalized=True)

    self.cnt = 0
  
  def infer_file(self, filepath=None, verbose=1):
    self.model.eval()

    audio, _ = torchaudio.load(filepath)

    if audio.shape[0] == 2:
      audio = audio[0][:]

    audio = self.transforms(audio).squeeze(0).transpose(0, 1)

    audio = [audio]

    audio = nn.utils.rnn.pad_sequence(audio, batch_first=True).unsqueeze(1).transpose(2, 3)

    output = self.model(audio.to(self.device))  
    output = F.log_softmax(output, dim=2)
    output = output.transpose(0, 1)

    decoded_preds = GreedyDecoder(output.transpose(0, 1).type(torch.IntTensor), mode='infer')

    if verbose == 1:
      print(f"File: {filepath} | Transcription: {decoded_preds[0]}")
    
    else:
      return decoded_preds[0]
  
  def infer_dir(self, directory=None):
    for root, _, files in os.walk(directory):
        for file in files:
          self.infer_file(os.path.join(root, file))
  
  def infer_stream(self):
    self.model.eval()

    print("Creating Audio stream...\nSpeak!")

    while True:
      try:
        self.stream = self.p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=48000,
                    input=True,
                    frames_per_buffer=1024)

        frames = []

        for _ in range(0, int(48000 / 1024 * 8)):
          data = np.frombuffer(self.stream.read(1024), dtype=np.float32)
          frames.append(data)
        
        frames = np.array(frames, dtype=np.float32).reshape(1, -1)
        
        audio = torch.from_numpy(frames)
        
        audio = self.transforms(audio).squeeze(0).transpose(0, 1)
        
        audio = [audio]
        audio = nn.utils.rnn.pad_sequence(audio, batch_first=True).unsqueeze(1).transpose(2, 3)

        output = self.model(audio.to(self.device)) 
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        decoded_preds = GreedyDecoder(output.transpose(0, 1).type(torch.IntTensor), mode='infer')

        print(''.join(decoded_preds), end=' ')
      
      except Exception as e:
        print(e)
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        break

      except KeyboardInterrupt:
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        del self.stream
        break
    
    self.stream.stop_stream()
    self.stream.close()
    self.p.terminate()
  
  def record_and_stream(self):
    self.model.eval()

    print("Creating Audio stream...\nSpeak!")

    self.stream = self.p.open(format=pyaudio.paInt16,
                    channels=2,
                    rate=48000,
                    input=True,
                    frames_per_buffer=1024)
         
    while True:
      try:
        frames = []

        for _ in range(0, int(48000 / 1024 * 6)):
          data = self.stream.read(1024)
          frames.append(data)
                  
        wf = wave.open(f'tamil_tech/__pycache__/{self.cnt}.wav', 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(48000)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(self.infer_file(f'tamil_tech/__pycache__/{self.cnt}.wav', verbose=False), end=' ')

      except Exception as e:
        print(e)
        break

      except KeyboardInterrupt:
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        break
    
    self.stream.stop_stream()
    self.stream.close()
    self.p.terminate()
  
  def __call__(self):
    self.record_and_stream()