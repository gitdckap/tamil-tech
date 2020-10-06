import os
import torch
import shutil
import torchvision
from tamil_tech.utils.torch_utils import *
from tamil_tech.models.torch import *
import torch.nn as nn
import torch.nn.functional as F

class TamilASR(object):
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

    output = self.model(audio.to(self.device))  # (batch, time, n_class)
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

        output = self.model(audio.to(self.device))  # (batch, time, n_class)
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

        # print("Opening Window...")

        for i in range(0, int(48000 / 1024 * 7)):
          data = self.stream.read(1024)
          frames.append(data)
                  
        wf = wave.open(f'tamil_tech/zero_shot/__pycache__/{self.cnt}.wav', 'wb')
        wf.setnchannels(2)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(48000)
        wf.writeframes(b''.join(frames))
        wf.close()

        # print("Closing Window...")

        print(self.infer_file(f'tamil_tech/zero_shot/__pycache__/{self.cnt}.wav', verbose=False), end=' ')

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