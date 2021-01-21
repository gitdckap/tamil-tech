# Tamil Tech

Tamil Tech is an Open-Source Library for end-to-end Natural Language Processing & Speech Processing for Tamil developed by DCKAP aiming to provide a common platform that allows developers, students, researchers, and experts around the world to build exciting AI based language and speech applications for Tamil language.

### Setup

#### From source 

* Clone the repository using the following command: `git clone https://github.com/gitdckap/tamil-tech`
* Navigate to the location of the repository where it was cloned on the machine: `cd tamil-tech`
* Install the library using: `pip install .` or `pip3 install .`

#### Using pip

The library can be installed using the [pip](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwif8Z7nr63uAhURfisKHVs0D7QQFjAAegQICBAC&url=https%3A%2F%2Fpypi.org%2Fproject%2Fpip%2F&usg=AOvVaw3bSOt-iIC9cvaKRbpvf4Yu) package manager for Python

`pip install tamil_tech`

## Documentation

Documentation for the library modules and functions have been specified in [Documentaion.md](tamil_tech/Documentation/md)

## Corpus & Pretrained Models

### Automatic Speech Recognition (ASR) Models (Speech To Text)

#### Speech Corpora

|               **Name**             |                                                  **Source**                                                         |
| :--------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
| OpenSLR                            |                                      [LibriSpeech](http://www.openslr.org/65)                                       |
| Microsoft Indian Languages (Tamil) | [Microsoft Speech Corpus (Indian languages)](https://msropendata.com/datasets/7230b4b1-912d-400e-be58-f84e0512985e) |
| Mozilla Common Voice - Tamil       |                      [https://commonvoice.mozilla.org](https://commonvoice.mozilla.org)                             |

#### Pretrained Models

Conformer S (character level) pretrained model is available at this [Drive link](https://drive.google.com/file/d/1thvXVQeqr0c-txBPEDvX8tNyniuo6SXW/view?usp=sharing)

| :-------------------------------------------------------------------------------------------------------------------: |
| **Model name:**                 | Conformer S                                                                         |
| **Vocabulary:**                 | Tamil unicode data (Available under `tamil_tech/vocabularies/tamil_unicode.txt`)    |
| **Test Word Error Rate (WER):** | 24.1 % (Beam Search), 42.8 % (Greedy)                                               |
| **Test Char Error Rate (WER):** | 10.2 % (Beam Search), 17.1 % (Greedy)                                               |
| :-------------------------------------------------------------------------------------------------------------------: |

## References

### Publications

* [Deep Speech](https://arxiv.org/abs/1412.5567.pdf)
* [Deep Speech 2](https://arxiv.org/pdf/1512.02595.pdf)
* [Deep Residual Learning For Image Recognition (ResNet paper for Residual Blocks)](https://arxiv.org/pdf/1512.03385.pdf)
* [Speech Transformer no-recurrence sequence to sequence](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005884.pdf)
* [RNN Transducer](https://arxiv.org/pdf/1211.3711.pdf)
* [Conformer Transducer](https://arxiv.org/pdf/2005.08100.pdf)
* [Streaming Transducer](https://arxiv.org/abs/1811.06621.pdf)
* [Sequence Transduction with Recurrent Neural Network](https://arxiv.org/abs/1211.3711)
* [Speech Transformer no-recurrence sequence to sequence](http://150.162.46.34:8080/icassp2018/ICASSP18_USB/pdfs/0005884.pdf)
* [ContextNet](http://arxiv.org/abs/2005.03191.pdf)
* [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
* [Self Attention With Relative Position Representations](https://arxiv.org/pdf/1803.02155v2.pdf)
* [Transformer-XL: Attentive Language Models Beyond a Fixed Length Context](https://arxiv.org/pdf/1901.02860.pdf)
* [Cold Fusion: Training Seq2Seq Models Together with Language Models](https://arxiv.org/pdf/1708.06426.pdf)
* [Mitsubishi's Streaming Speech Recognition Transformer](https://www.merl.com/publications/docs/TR2020-040.pdf)
* [Time Restricted Self Attention](https://www.danielpovey.com/files/2018_icassp_attention.pdf)
* [Triggered Attention](https://www.merl.com/publications/docs/TR2019-015.pdf)

### Code & Repositories

1. [Tensorflow ASR](https://github.com/TensorSpeech/TensorFlowASR)
2. [NVIDIA OpenSeq2Seq Toolkit](https://github.com/NVIDIA/OpenSeq2Seq)
3. [Facebook's Fairseq](https://github.com/pytorch/fairseq)
4. [TF2 Speech Reocgnition Transformer](https://github.com/YoungloLee/tf2-speech-recognition-transformer)
5. [WARP Transducer](https://github.com/noahchalifour/warp-transducer)
6. [Open Speech Corpora](https://github.com/JRMeyer/open-speech-corpora)
7. [Awni Speech](https://github.com/awni/speech)
8. [Sequence Transduction with Recurrent Neural Network](https://arxiv.org/abs/1211.3711)
9. [End-to-End Speech Processing Toolkit in PyTorch](https://github.com/espnet/espnet)
10. [https://github.com/iankur/ContextNet](https://github.com/iankur/ContextNet)

## Contributors

* [Akshay Kumaar M](https://github.com/aksh-ai)

## About

An awesome open-source initiative by DCKAP and DCKAP's Artificial Intelligence Team.

![DCKAP Logo](images/DCKAP-Organization-Logo.png)
