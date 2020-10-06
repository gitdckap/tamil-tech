import os
import abc
import math
import collections
import tensorflow as tf
from tamil_tech.encoders import *
from tamil_tech.layers.tf import *
from tamil_tech.utils.tf_utils import *
from tamil_tech.utils.featurizers.speech_featurizer import TFSpeechFeaturizer
from tamil_tech.utils.featurizers.text_featurizer import TextFeaturizer
from ctc_decoders import ctc_greedy_decoder, ctc_beam_search_decoder

Hypothesis = collections.namedtuple(
    "Hypothesis",
    ("index", "prediction", "states")
)

BeamHypothesis = collections.namedtuple(
    "BeamHypothesis",
    ("score", "prediction", "states", "lm_states")
)

class Model(tf.keras.Model):
    def __init__(self, name, **kwargs):
        super(Model, self).__init__(name=name, **kwargs)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()

class CtcModel(tf.keras.Model):
    def __init__(self,
                 base_model: tf.keras.Model,
                 num_classes: int,
                 name="ctc_model",
                 **kwargs):
        super(CtcModel, self).__init__(name=name, **kwargs)
        self.base_model = base_model
        # Fully connected layer
        self.fc = tf.keras.layers.Dense(units=num_classes, activation="linear",
                                        use_bias=True, name=f"{name}_fc")

    def _build(self, input_shape):
        features = tf.keras.Input(input_shape, dtype=tf.float32)
        self(features, training=False)

    def summary(self, line_length=None, **kwargs):
        self.base_model.summary(line_length=line_length, **kwargs)
        super(CtcModel, self).summary(line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: TFSpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False, **kwargs):
        outputs = self.base_model(inputs, training=training)
        outputs = self.fc(outputs, training=training)
        return outputs

    def get_config(self):
        config = self.base_model.get_config()
        config.update(self.fc.get_config())
        return config

    # Greedy

    @tf.function
    def recognize(self, features):
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob):
            return tf.numpy_function(self.perform_greedy,
                                     inp=[prob], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def perform_greedy(self, probs: np.ndarray):
        decoded = ctc_greedy_decoder(probs, vocabulary=self.text_featurizer.vocab_array)
        return tf.convert_to_tensor(decoded, dtype=tf.string)

    def recognize_tflite(self, signal):
        """
        Function to convert to tflite using greedy decoding
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
        """
        features = self.speech_featurizer.tf_extract(signal)
        features = tf.expand_dims(features, axis=0)
        input_length = shape_list(features)[1]
        input_length = input_length // self.base_model.time_reduction_factor
        input_length = tf.expand_dims(input_length, axis=0)
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)
        decoded = tf.keras.backend.ctc_decode(
            y_pred=probs, input_length=input_length, greedy=True
        )
        decoded = tf.cast(decoded[0][0][0], dtype=tf.int32)
        transcript = self.text_featurizer.indices2upoints(decoded)
        return transcript

    # Beam Search

    @tf.function
    def recognize_beam(self, features, lm=False):
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)

        def map_fn(prob):
            return tf.numpy_function(self.perform_beam_search,
                                     inp=[prob, lm], Tout=tf.string)

        return tf.map_fn(map_fn, probs, dtype=tf.string)

    def perform_beam_search(self,
                            probs: np.ndarray,
                            lm: bool = False):
        decoded = ctc_beam_search_decoder(
            probs_seq=probs,
            vocabulary=self.text_featurizer.vocab_array,
            beam_size=self.text_featurizer.decoder_config["beam_width"],
            ext_scoring_func=self.text_featurizer.scorer if lm else None
        )
        decoded = decoded[0][-1]

        return tf.convert_to_tensor(decoded, dtype=tf.string)

    def recognize_beam_tflite(self, signal):
        features = self.speech_featurizer.tf_extract(signal)
        features = tf.expand_dims(features, axis=0)
        input_length = shape_list(features)[1]
        input_length = input_length // self.base_model.time_reduction_factor
        input_length = tf.expand_dims(input_length, axis=0)
        logits = self(features, training=False)
        probs = tf.nn.softmax(logits)
        decoded = tf.keras.backend.ctc_decode(
            y_pred=probs, input_length=input_length, greedy=False,
            beam_width=self.text_featurizer.decoder_config["beam_width"]
        )
        decoded = tf.cast(decoded[0][0][0], dtype=tf.int32)
        transcript = self.text_featurizer.indices2upoints(decoded)
        return transcript

    # TFLite

    def make_tflite_function(self, greedy: bool = False):
        if greedy:
            return tf.function(
                self.recognize_tflite,
                input_signature=[
                    tf.TensorSpec([None], dtype=tf.float32)
                ]
            )
        return tf.function(
            self.recognize_beam_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32)
            ]
        )

class TransducerPrediction(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 512,
                 rnn_type: str = "lstm",
                 layer_norm=True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = Embedding(vocabulary_size, embed_dim,
                               regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        # Initialize rnn layers
        RNN = get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units=rnn_units, return_sequences=True,
                name=f"{name}_lstm_{i}", return_state=True,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            if layer_norm:
                ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}")
            else:
                ln = None
            self.rnns.append({"rnn": rnn, "ln": ln})

    def get_initial_state(self):
        """Get zeros states
        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(
                tf.stack(
                    rnn["rnn"].get_initial_state(
                        tf.zeros([1, 1, 1], dtype=tf.float32)
                    ), axis=0
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False):
        # inputs has shape [B, U]
        # use tf.gather_nd instead of tf.gather for tflite conversion
        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        for rnn in self.rnns:
            outputs = rnn["rnn"](outputs, training=training)
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for prediction network
        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]
        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, training=False,
                                 initial_state=tf.unstack(states[i], axis=0))
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.embed.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn["rnn"].get_config())
            if rnn["ln"] is not None:
                conf.update(rnn["ln"].get_config())
        return conf

class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="tranducer_joint",
                 **kwargs):
        super(TransducerJoint, self).__init__(name=name, **kwargs)
        self.ffn_enc = tf.keras.layers.Dense(
            joint_dim, name=f"{name}_enc",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ffn_pred = tf.keras.layers.Dense(
            joint_dim, use_bias=False, name=f"{name}_pred",
            kernel_regularizer=kernel_regularizer
        )
        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size, name=f"{name}_vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    def call(self, inputs, training=False):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
        pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        enc_out = tf.expand_dims(enc_out, axis=2)
        pred_out = tf.expand_dims(pred_out, axis=1)
        outputs = tf.nn.tanh(enc_out + pred_out)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        return conf

class Transducer(Model):
    """ Transducer Model Warper """
    def __init__(self,
                 encoder: tf.keras.Model,
                 vocabulary_size: int,
                 embed_dim: int = 512,
                 embed_dropout: float = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 320,
                 rnn_type: str = "lstm",
                 layer_norm: bool = True,
                 joint_dim: int = 1024,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="transducer",
                 **kwargs):
        super(Transducer, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_rnns=num_rnns,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            layer_norm=layer_norm,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_prediction"
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_joint"
        )

    def _build(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        self([inputs, pred], training=False)

    def summary(self, line_length=None, **kwargs):
        if self.encoder is not None:
            self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: TFSpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            predicted: predicted sequence of character ids, in shape [B, U]
            training: python boolean
            **kwargs: sth else
        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, predicted = inputs
        enc = self.encoder(features, training=training)
        pred = self.predict_net(predicted, training=training)
        outputs = self.joint_net([enc, pred], training=training)
        return outputs

    def encoder_inference(self, features):
        """Infer function for encoder (or encoders)
        Args:
            features (tf.Tensor): features with shape [T, F, C]
        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs = self.encoder(outputs, training=False)
            return tf.squeeze(outputs, axis=0)

    def decoder_inference(self, encoded, predicted, states):
        """Infer function for decoder
        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers
        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predict_net.recognize(predicted, states)  # [1, 1, P], states
            ytu = tf.nn.log_softmax(self.joint_net([encoded, y], training=False))  # [1, 1, V]
            ytu = tf.squeeze(ytu, axis=None)  # [1, 1, V] => [V]
            return ytu, new_states

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # greedy

    @tf.function
    def recognize(self, features):
        total = tf.shape(features)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, features, decoded): return tf.less(batch, total)

        def body(batch, total, features, decoded):
            yseq = self.perform_greedy(
                features[batch],
                predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                states=self.predict_net.get_initial_state(),
                swap_memory=True
            )
            yseq = self.text_featurizer.iextract(tf.expand_dims(yseq.prediction, axis=0))
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, features, decoded

        batch, total, features, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, features, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(features),
                tf.TensorShape([None])
            )
        )

        return decoded

    def recognize_tflite(self, signal, predicted, states):
        """
        Function to convert to tflite using greedy decoding (default streaming mode)
        Args:
            signal: tf.Tensor with shape [None] indicating a single audio signal
            predicted: last predicted character with shape []
            states: lastest rnn states with shape [num_rnns, 1 or 2, 1, P]
        Return:
            transcript: tf.Tensor of Unicode Code Points with shape [None] and dtype tf.int32
            predicted: last predicted character with shape []
            states: lastest rnn states with shape [num_rnns, 1 or 2, 1, P]
        """
        features = self.speech_featurizer.tf_extract(signal)
        hypothesis = self.perform_greedy(features, predicted, states, swap_memory=False)
        transcript = self.text_featurizer.indices2upoints(hypothesis.prediction)
        return (
            transcript,
            hypothesis.prediction[-1],
            hypothesis.states
        )

    def perform_greedy(self, features, predicted, states, swap_memory=False):
        with tf.name_scope(f"{self.name}_greedy"):
            encoded = self.encoder_inference(features)
            # Initialize prediction with a blank
            # Prediction can not be longer than the encoded of audio plus blank
            prediction = tf.TensorArray(
                dtype=tf.int32,
                size=(tf.shape(encoded)[0] + 1),
                dynamic_size=False,
                element_shape=tf.TensorShape([]),
                clear_after_read=False
            )
            time = tf.constant(0, dtype=tf.int32)
            total = tf.shape(encoded)[0]

            hypothesis = Hypothesis(
                index=tf.constant(0, dtype=tf.int32),
                prediction=prediction.write(0, predicted),
                states=states
            )

            def condition(time, total, encoded, hypothesis): return tf.less(time, total)

            def body(time, total, encoded, hypothesis):
                ytu, new_states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.expand_dims(time, axis=-1)),
                    predicted=hypothesis.prediction.read(hypothesis.index),
                    states=hypothesis.states
                )
                char = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                index, char, new_states = tf.cond(
                    tf.equal(char, self.text_featurizer.blank),
                    true_fn=lambda: (
                        hypothesis.index,
                        hypothesis.prediction.read(hypothesis.index),
                        hypothesis.states
                    ),
                    false_fn=lambda: (
                        hypothesis.index + 1,
                        char,
                        new_states
                    )
                )

                hypothesis = Hypothesis(
                    index=index,
                    prediction=hypothesis.prediction.write(index, char),
                    states=new_states
                )

                return time + 1, total, encoded, hypothesis

            time, total, encoded, hypothesis = tf.while_loop(
                condition,
                body,
                loop_vars=(time, total, encoded, hypothesis),
                swap_memory=swap_memory
            )

            # Gather predicted sequence
            hypothesis = Hypothesis(
                index=hypothesis.index,
                prediction=tf.gather_nd(
                    params=hypothesis.prediction.stack(),
                    indices=tf.expand_dims(tf.range(hypothesis.index + 1), axis=-1)
                ),
                states=hypothesis.states
            )

            return hypothesis

    # beam search

    @tf.function
    def recognize_beam(self, features, lm=False):
        total = tf.shape(features)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, features, decoded): return tf.less(batch, total)

        def body(batch, total, features, decoded):
            yseq = tf.py_function(self.perform_beam_search,
                                  inp=[features[batch], lm],
                                  Tout=tf.int32)
            yseq = self.text_featurizer.iextract(yseq)
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, features, decoded

        batch, total, features, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, features, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(features),
                tf.TensorShape([None]),
            )
        )

        return decoded

    def perform_beam_search(self, features, lm=False):
        beam_width = self.text_featurizer.decoder_config["beam_width"]
        norm_score = self.text_featurizer.decoder_config["norm_score"]
        lm = lm.numpy()

        kept_hyps = [
            BeamHypothesis(
                score=0.0,
                prediction=[self.text_featurizer.blank],
                states=self.predict_net.get_initial_state(),
                lm_states=None
            )
        ]

        enc = self.encoder_inference(features)
        total = tf.shape(enc)[0].numpy()

        B = kept_hyps

        for i in range(total):  # [E]
            A = B  # A = hyps
            B = []

            while True:
                y_hat = max(A, key=lambda x: x.score)
                A.remove(y_hat)

                ytu, new_states = self.decoder_inference(
                    encoded=tf.gather_nd(enc, tf.expand_dims(i, axis=-1)),
                    predicted=y_hat.prediction[-1],
                    states=y_hat.states
                )

                if lm and self.text_featurizer.scorer:
                    lm_state, lm_score = self.text_featurizer.scorer(y_hat)

                for k in range(self.text_featurizer.num_classes):
                    beam_hyp = BeamHypothesis(
                        score=(y_hat.score + float(ytu[k].numpy())),
                        prediction=y_hat.prediction,
                        states=y_hat.states,
                        lm_states=y_hat.lm_states
                    )

                    if k == self.text_featurizer.blank:
                        B.append(beam_hyp)
                    else:
                        beam_hyp = BeamHypothesis(
                            score=beam_hyp.score,
                            prediction=(beam_hyp.prediction + [int(k)]),
                            states=new_states,
                            lm_states=beam_hyp.lm_states
                        )

                        if lm and self.text_featurizer.scorer:
                            beam_hyp = BeamHypothesis(
                                score=(beam_hyp.score + lm_score),
                                prediction=beam_hyp.prediction,
                                states=new_states,
                                lm_states=lm_state
                            )

                        A.append(beam_hyp)

                if len(B) > beam_width: break

        if norm_score:
            kept_hyps = sorted(B, key=lambda x: x.score / len(x.prediction),
                               reverse=True)[:beam_width]
        else:
            kept_hyps = sorted(B, key=lambda x: x.score, reverse=True)[:beam_width]

        return tf.convert_to_tensor(kept_hyps[0].prediction, dtype=tf.int32)[None, ...]

    def make_tflite_function(self, greedy: bool = True):
        return tf.function(
            self.recognize_tflite,
            input_signature=[
                tf.TensorSpec([None], dtype=tf.float32),
                tf.TensorSpec([], dtype=tf.int32),
                tf.TensorSpec(self.predict_net.get_initial_state().get_shape(),
                              dtype=tf.float32)
            ]
        )

class Conformer(Transducer):
    def __init__(self,
                 subsampling: dict,
                 positional_encoding: str = "sinusoid",
                 dmodel: int = 144,
                 vocabulary_size: int = 29,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 mha_type: str = "relmha",
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 320,
                 rnn_type: str = "lstm",
                 layer_norm: bool = True,
                 joint_dim: int = 1024,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name: str = "conformer_transducer",
                 **kwargs):
        super(Conformer, self).__init__(
            encoder=ConformerEncoder(
                subsampling=subsampling,
                positional_encoding=positional_encoding,
                dmodel=dmodel,
                num_blocks=num_blocks,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                fc_factor=fc_factor,
                dropout=dropout,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            ),
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_rnns=num_rnns,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            layer_norm=layer_norm,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name, **kwargs
        )
        self.time_reduction_factor = self.encoder.conv_subsampling.time_reduction_factor

class MultiConformers(Transducer):
    def __init__(self,
                 subsampling: dict,
                 positional_encoding: str = "sinusoid",
                 dmodel: int = 144,
                 vocabulary_size: int = 29,
                 num_blocks: int = 16,
                 head_size: int = 36,
                 num_heads: int = 4,
                 mha_type: str = "relmha",
                 kernel_size: int = 32,
                 fc_factor: float = 0.5,
                 dropout: float = 0,
                 embed_dim: int = 512,
                 embed_dropout: int = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 320,
                 rnn_type: str = "lstm",
                 layer_norm: bool = True,
                 encoder_joint_mode: str = "concat",
                 joint_dim: int = 1024,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="multi-conformers",
                 **kwargs):
        super(MultiConformers, self).__init__(
            encoder=None,
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_rnns=num_rnns,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            layer_norm=layer_norm,
            joint_dim=joint_dim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=name,
        )
        self.encoder_lms = ConformerEncoder(
            subsampling=subsampling,
            positional_encoding=positional_encoding,
            dmodel=dmodel,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            mha_type=mha_type,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder_lms"
        )
        self.encoder_lgs = ConformerEncoder(
            subsampling=subsampling,
            positional_encoding=positional_encoding,
            dmodel=dmodel,
            num_blocks=num_blocks,
            head_size=head_size,
            num_heads=num_heads,
            mha_type=mha_type,
            kernel_size=kernel_size,
            fc_factor=fc_factor,
            dropout=dropout,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_encoder_lgs"
        )
        if encoder_joint_mode == "concat":
            self.encoder_joint = Concatenation(dmodel, name=f"{name}_enc_concat")
        elif encoder_joint_mode == "add":
            self.encoder_joint = tf.keras.layers.Add(name=f"{name}_enc_add")
        else:
            raise ValueError("encoder_joint_mode must be either 'concat' or 'add'")
        self.time_reduction_factor = self.encoder_lms.conv_subsampling.time_reduction_factor

    def _build(self, lms_shape, lgs_shape):
        lms = tf.keras.Input(shape=lms_shape, dtype=tf.float32)
        lgs = tf.keras.Input(shape=lgs_shape, dtype=tf.float32)
        pred = tf.keras.Input(shape=[None], dtype=tf.int32)
        self([lms, lgs, pred], training=False)

    def summary(self, line_length=None, **kwargs):
        self.encoder_lms.summary(line_length=line_length, **kwargs)
        super(MultiConformers, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer_lms: SpeechFeaturizer,
                        speech_featurizer_lgs: SpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tflite
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer_lms = speech_featurizer_lms
        self.speech_featurizer_lgs = speech_featurizer_lgs
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False):
        lms, lgs, predicted = inputs

        pred = self.predict_net(predicted, training=training)

        enc_lms_out = self.encoder_lms(lms, training=training)
        outputs_lms = self.joint_net([enc_lms_out, pred], training=training)

        enc_lgs_out = self.encoder_lgs(lgs, training=training)
        outputs_lgs = self.joint_net([enc_lgs_out, pred], training=training)

        enc = self.encoder_joint([enc_lms_out, enc_lgs_out], training=training)
        outputs = self.joint_net([enc, pred], training=training)

        return outputs_lms, outputs, outputs_lgs

    def encoder_inference(self, features):
        """Infer function for encoder (or encoders)
        Args:
            features (list or tuple): 2 features, each has shape [T, F, C]
        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            lms, lgs = features
            lms = tf.expand_dims(lms, axis=0)
            lgs = tf.expand_dims(lgs, axis=0)

            enc_lms_out = self.encoder_lms(lms, training=False)
            enc_lgs_out = self.encoder_lgs(lgs, training=False)

            outputs = self.encoder_joint([enc_lms_out, enc_lgs_out], training=False)

            return tf.squeeze(outputs, axis=0)

    def get_config(self):
        conf = self.encoder_lms.get_config()
        conf.update(self.encoder_lgs.get_config())
        conf.update(self.concat.get_config())
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # Greedy

    @tf.function
    def recognize(self, features):
        lms, lgs = features

        total = tf.shape(lms)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, lms, lgs, decoded): return tf.less(batch, total)

        def body(batch, total, lms, lgs, decoded):
            yseq = self.perform_greedy(
                [lms[batch], lgs[batch]],
                predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                states=self.predict_net.get_initial_state(),
                swap_memory=True
            )
            yseq = self.text_featurizer.iextract(tf.expand_dims(yseq.prediction, axis=0))
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, lms, lgs, decoded

        batch, total, lms, lgs, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, lms, lgs, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(lms),
                get_shape_invariants(lgs),
                tf.TensorShape([None])
            )
        )

        return decoded

    # Beam Search

    @tf.function
    def recognize_beam(self, features, lm=False):
        lms, lgs = features

        total = tf.shape(lms)[0]
        batch = tf.constant(0, dtype=tf.int32)

        decoded = tf.constant([], dtype=tf.string)

        def condition(batch, total, lms, lgs, decoded): return tf.less(batch, total)

        def body(batch, total, lms, lgs, decoded):
            yseq = tf.py_function(self.perform_beam_search,
                                  inp=[[lms[batch], lgs[batch]], lm],
                                  Tout=tf.int32)
            yseq = self.text_featurizer.iextract(yseq)
            decoded = tf.concat([decoded, yseq], axis=0)
            return batch + 1, total, lms, lgs, decoded

        batch, total, lms, lgs, decoded = tf.while_loop(
            condition,
            body,
            loop_vars=(batch, total, lms, lgs, decoded),
            swap_memory=True,
            shape_invariants=(
                batch.get_shape(),
                total.get_shape(),
                get_shape_invariants(lms),
                get_shape_invariants(lgs),
                tf.TensorShape([None]),
            )
        )

        return decoded