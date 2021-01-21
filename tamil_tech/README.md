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

The Layers module consists of multiple activation and neural network layers as torch modules, mostly taken from PyTorch forums and research papers. Apart from these, PyTorch's layers can be used and different version of layers for the same are given in this module.

#### CustomSwish

Swish activation function with foward and backward pass for memory and computational optimization.

**Methods:**

* **forward:** Performs forward pass of Swish over input tensor and returns swish activation applied output tensor
* **backward:** Performs backward pass of Swish over input tensor and returns the gradient

#### Swish

Swish activation function that can be used as a layer

**Args:** Accepts input tensor and applies swish activation over it

#### SpecAug

Spectrogram Augmentation as a layer.

**Args**:

* training: bool - If training or not, for applying spectorgram augmentation
* freq_len: int - Frequency Length
* time_len: int - Time length
* freq_mask: int - Frequency masking parameter
* time_mask: int - Time masking parameter

**Members:**

* training: bool - If training or not, for applying spectorgram augmentation
* freq_len: int - Frequency Length
* time_len: int - Time length
* freq_mask: int - Frequency masking parameter
* time_mask: int - Time masking parameter

**Methods:**

* **forward:** Applies frequency masking and time masking over the input tensor and returns augmented spectrogram

#### LayerNorm & LayerNormaliztion

Layer Normalization with batch and time transposed

#### ResidualBlock & Residual

ResidualNet based feature extraction as a layer

**Args**:

* in_channels: int - input channels
* out_channels: int - output channels
* kernel: int - Size of kernel
* stride: int - Size of stride
* n_feats: int - Number of features for normalization
* dropout: float - dropout rate

**Members:**

* ln1, ln2 - Layer Normalziation
* dropout1, dropout2 - Dropout Layer
* cnn1, cnn2 - Convolution 2D Layer
* swish1, swish2 - Swish Activation
* bn1, bn2 - Batch Normalization 2D

**Methods:**

* **forward:** returns residual tensor added with feature extracted tensor

#### BidirectionalRNN & BiGRU

**Args**:

* rnn_dim: int - Dimensions for RNN layer
* hidden_size: int - Hidden units
* batch_first: bool - batch as first axis in tensor or not
* dropout: float - dropout rate

* rnn - Type of RNN to be used, GRU or LSTM

**Members:**

* layer_norm - Layer Normalization
* BiGRU - bidirectional RNN
* swish - swish activation
* dropout - dropout layer

**Methods:**

* **forward:** returns seqeunce tensor

### Encoders

Encoders module consists of multiple encoders that can be used as front-end encoder for encoding or feature-extraction in the transformer based network.

#### EncoderLayer

Composed with two sub-layers: A multi-head self-attention mechanism, A simple, position-wise fully connected feed-forward network.

**Args:**

* d_model - number of dimensions for model
* d_inner - nummber of dimensions for inner 
* n_head - number of attention heads
* d_k - number of dimensions for key
* d_v - number of dimensions for value
* dropout - dropout rate

**Members:**

* slf_attn - Self attention layer
* pos_ffn - Positional Feed Forward layer

**Methods:**

* forward: Performs decoding operations via the layers and returns decoded tensor, and self attention weights

#### Encoder

**Args:**

* d_input - input dimensions
* n_layers - number of layers
* n_head - number of attention heads
* d_k - number of dimensions for key
* d_v - number of dimensions for value
* d_model - number of dimensions for model
* d_inner - nummber of dimensions for inner 
* dropout - dropout rate
* pe_maxlen - positional encoding maximum length

**Members:**

* d_input - input dimensions
* n_layers - number of layers
* n_head - number of attention heads
* d_k - number of dimensions for key
* d_v - number of dimensions for value
* d_model - number of dimensions for model
* d_inner - nummber of dimensions for inner 
* dropout_rate - dropout rate
* pe_maxlen - positional encoding maximum length
* linear_in - Linear layer for input embedding
* layer_norm_in - Layer Normalization for input embedding
* positional_encoding - positional encoding layer
* dropout - dropout layer
* layer_stack - Encoder Layers as a list

**Methods**

* **forward:** Accepts padded input, length of inputs and performs the encoding operation and returns output tensor and attention weights if selected as True.

### Decoders

Decoders module consists of multiple decoders that can be used for decoding operations in the neural network.

### DecoderLayer

**Args:** 

* d_model - number of dimensions for model
* d_inner - nummber of dimensions for inner 
* n_head - number of attention heads
* d_k - number of dimensions for key
* d_v - number of dimensions for value
* dropout - dropout rate

**Members:**

* slf_attn - Self attention layer
* enc_attn - Self attention layer for encoding
* pos_ffn - Positional Feed Forward layer

**Methods:**

* forward: Performs decoding operations via the layers and returns decoded tensor, self attention weights and encoded attention weights

#### Decoder

A self attention based decoder

**Args:** 

* sos_id - Start of Sentence ID
* eod_id - End of Sentence ID
* n_tgt_vocab - target vocab size
* d_word_vec - word vectore dimensions
* n_layers - number of layers
* n_head - number of attention heads
* d_k - number of dimensions for key
* d_v - number of dimensions for value
* d_model - number of dimensions for model
* d_inner - nummber of dimensions for inner 
* dropout - dropout rate
* tgt_emb_prj_weight_sharing - target embedding weight sharing
* pe_maxlen - positional encoding maximum length

**Members:**

* sos_id - Start of Sentence ID
* eod_id - End of Sentence ID
* n_tgt_vocab - target vocab size
* d_word_vec - word vectore dimensions
* n_layers - number of layers
* n_head - number of attention heads
* d_k - number of dimensions for key
* d_v - number of dimensions for value
* d_model - number of dimensions for model
* d_inner - nummber of dimensions for inner 
* dropout - dropout rate
* tgt_emb_prj_weight_sharing - target embedding weight sharing
* pe_maxlen - positional embedding maximum length
* tgt_word_emb - target embedding layer
* positional encoding - positional encoding layer
* dropout - dropout layer
* layer_stack - List of Decoder layers
* tgt_word_prj - Target word projection Linear layer

**Methods:**

* **preporcess:** Generates decoder input and output label from padded_input. Add <sos> to decoder input, and add <eos> to decoder output label. Returns padded input text and labels
* **forward:** Processes the decoding network and returns decoded output tensor
* **recognize_beam:** Performs Beam Search, decodes single utterance

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