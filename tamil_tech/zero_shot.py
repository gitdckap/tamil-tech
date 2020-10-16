import os
import io
import torch
import shutil
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tamil_tech.torch.utils import *
from tamil_tech.torch.models import *

import librosa
import soundfile as sf
import tensorflow as tf
from tamil_tech.tf.utils.loaders import load_conformer_model

class ConformerTamilASR(object):
  # bug - segmentation fault during TFLite inference
  def __init__(self, greedy=False, use_gpu=False, tflite=True, conformer_type='M'):
    if use_gpu == False:
      try:
          # Disable all GPUS
          tf.config.set_visible_devices([], 'GPU')
          visible_devices = tf.config.get_visible_devices()
          for device in visible_devices:
              assert device.device_type != 'GPU'
          os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
      except:
          # Invalid device or cannot modify virtual devices once initialized.
          pass
          
    self.tflite = tflite
    self.greedy = greedy

    self.model, self.state_size, self.blank = load_conformer_model(tflite=self.tflite, greedy=self.greedy, conformer_type=conformer_type)

    if self.model:
      print("Loaded Model!")

    if self.tflite:
      self.input_details = self.model.get_input_details()
      self.output_details = self.model.get_output_details()
  
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
    
  def infer_file(self, audio, sample_rate=16000):
    signal = self.read_raw_audio(audio=audio, sample_rate=sample_rate)

    if self.tflite:
      self.model.resize_tensor_input(self.input_details[0]["index"], signal.shape)
      self.model.allocate_tensors()
      self.model.set_tensor(self.input_details[0]["index"], signal)
      
      self.model.set_tensor(
          self.input_details[1]["index"],
          tf.constant(self.blank, dtype=tf.int32)
      )
      
      self.model.set_tensor(
          self.input_details[2]["index"],
          tf.zeros([1, 2, 1, self.state_size], dtype=tf.float32)
      )
      
      _ = self.model.invoke()
      
      predictions = self.model.get_tensor(self.output_details[0]["index"])
      
      print("".join([chr(u) for u in predictions]))
    
    else:
      signal = np.expand_dims(self.model.speech_featurizer.extract(signal), axis=1)
      if self.greedy:
        pred = self.model.recognize(signal)
      
      if self.model.text_featurizer.decoder_config["beam_width"] > 0 and not self.greedy:
          pred = self.model.recognize_beam(features=signal, lm=True)
      else:
          pred = beam_lm_pred = tf.constant([""], dtype=tf.string)

      print(tf.convert_to_tensor([pred], dtype=tf.string))
  
  def infer_dir(self, directory=None):
    for root, dirs, files in os.walk(directory):
        for file in files:
          self.infer_file(os.path.join(root, file))
  
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
        transcription = ""
        audios = []
        frames = []

        for i in range(0, int(16000 / 1024 * 6)):
          data = self.stream.read(1024)
          frames.append(data)
                  
        wf = wave.open(f'tamil_tech/__pycache__/{self.cnt}.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.infer_file(f'tamil_tech/__pycache__/{self.cnt}.wav')

      except Exception as e:
        print(e)
        break

      except KeyboardInterrupt:
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        break
  
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

    audio, sr = torchaudio.load(filepath)

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
    for root, dirs, files in os.walk(directory):
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

        transcription = ""
        
        audios = []
        frames = []

        for i in range(0, int(48000 / 1024 * 8)):
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
        transcription = ""
        audios = []
        frames = []

        for i in range(0, int(48000 / 1024 * 6)):
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
  
  def __call__(self):
    self.record_and_stream()