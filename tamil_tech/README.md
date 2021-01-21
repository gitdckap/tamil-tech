# Documentation

## Configs

Configuration files for the ASR models as per [TensorflowASR](https://github.com/TensorSpeech/TensorFlowASR)'s documentation

* conformer_new_config.yml - Conformer S (Character level) config file for Tamil ASR Model
* subword_conformer_new.yml - Conformer S (Subwords/N-grams) config file for Tamil ASR Model

## Vocabularies

All these vocabulary files have been created as per [TensorflowASR](https://github.com/TensorSpeech/TensorFlowASR)'s documentation

* english.txt - Vocabulary file for English language
* tamil_new_vocab.txt - Vocabulary file for Tamil Language without Vada sorkal (Sanskrit and Hindi based letters)
* tamil_tokenized_v3.json - Vocabulary file for Tamil Language and PyTorch ASR Model without Vada sorkal (Sanskrit and Hindi based letters)
* tamil_unicode.txt - Vocabulary file for Tamil Language with Vada sorkal (Sanskrit and Hindi based letters)

## Torch

All the modules specified under this header are for PyTorch based ASR and language models

### Datasets

The datasets module that contains classes that can download datasets for ASR as well as Language models onto user's computer automatically and can be used to spawn necessary data using an iterator (a dataset pipeline)

### Layers

### Encoders

### Decoders

### Models

### Loss

### Optimizers

### Run

### Transforms

### Utils

## Zero Shot

The zero shot feature of Tamil Tech allows developers to implements straight-forward pipelines for Speech to Text, Automatic Speech Recognition, Language models based text generation, etc with just 2-5 lines of code. The zero shot module consists of 4 classes:

### ConformerTamilASR

The Conformer Tamil ASR class is the ASR model that uses character level prediction for Speech To Text. It downloads pretrained models directly onto the user's computer if it is not already present.

#### Args

**path : string** - Accepts the path to the weights file (.h5 file) of the Conformer S pretrained model. If the model is not present in the specified path, then the model is downloaded automatically.

#### Members

**model** - The Conformer S model containing pretrained weights, text featurizer, and speech featurizer for decoding sequence to text and converting audio signal to spectrogram respectively.

#### Methods

* **read_raw_audio:** Reads raw audio and returns audio signal as a numpy array. Accepts audio file path as string, bytes, or numpy array
* **bytes_to-string:** Returns decoded string of the text sequence with unicode encoding with vocabulary. Accepts Tensor or Numpy array as argument.
* **infer:** Returns the decoded string of predicted text sequence from the audio file, bytes passed to the function.
* **inder_dir:** Prints the predicted text sequence for audio files presnent in the directory path passed to the function.

### SubwordConformerTamilASR

The Conformer Tamil ASR class is the ASR model that uses subwords or N-grams (4 characters) prediction for Speech To Text. It downloads pretrained models directly onto the user's computer if it is not already present.

#### Args

**path : string** - Accepts the path to the weights file (.h5 file) of the Conformer S pretrained model. If the model is not present in the specified path, then the model is downloaded automatically.

#### Members

**model** - The Conformer S model containing pretrained weights, text featurizer, and speech featurizer for decoding sequence to text and converting audio signal to spectrogram respectively.

#### Methods

* **read_raw_audio:** Reads raw audio and returns audio signal as a numpy array. Accepts audio file path as string, bytes, or numpy array
* **bytes_to-string:** Returns decoded string of the text sequence with unicode encoding with vocabulary. Accepts Tensor or Numpy array as argument.
* **infer:** Returns the decoded string of predicted text sequence from the audio file, bytes passed to the function.
* **inder_dir:** Prints the predicted text sequence for audio files presnent in the directory path passed to the function.

### ConformerTamilASRLite

The Conformer Tamil ASR class is the ASR model that uses character level prediction for Speech To Text with '.tflite' model. It downloads pretrained models directly onto the user's computer if it is not already present.

#### Args

**path : string** - Accepts the path to the tflite file (.tflite file) of the Conformer S pretrained model. If the model is not present in the specified path, then the model is downloaded automatically.

#### Members

**model** - The Conformer S model containing pretrained weights, text featurizer, and speech featurizer for decoding sequence to text and converting audio signal to spectrogram respectively.

#### Methods

* **read_raw_audio:** Reads raw audio and returns audio signal as a numpy array. Accepts audio file path as string, bytes, or numpy array
* **infer:** Returns the decoded string of predicted text sequence from the audio file, bytes passed to the function.
* **inder_dir:** Prints the predicted text sequence for audio files presnent in the directory path passed to the function.

### DeepTamilASR

The Deep Tamil ASR class is the ASR model similar to Deep Speech 2 architecture, that uses character level (4 characters) prediction for Speech To Text. It downloads pretrained models directly onto the user's computer if it is not already present.

#### Args

**use_gpu : bool** - Specifiy if the model should use GPU or not, for computation.

#### Members

**model** - The Deep Tmail ASR model containing pretrained weights.

#### Methods

* **infer:** Returns the decoded string of predicted text sequence from the audio file passed to the function.
* **inder_dir:** Prints the predicted text sequence for audio files presnent in the directory path passed to the function.