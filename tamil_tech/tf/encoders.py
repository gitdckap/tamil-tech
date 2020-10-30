import tensorflow as tf
from tamil_tech.tf.layers import *

class ConformerEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 subsampling,
                 positional_encoding="sinusoid",
                 dmodel=144,
                 num_blocks=16,
                 mha_type="relmha",
                 head_size=36,
                 num_heads=4,
                 kernel_size=32,
                 depth_multiplier=1,
                 fc_factor=0.5,
                 dropout=0.0,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conformer_encoder",
                 **kwargs):
        super(ConformerEncoder, self).__init__(name=name, **kwargs)

        subsampling_name = subsampling.pop("type", "conv2d")
        if subsampling_name == "vgg":
            subsampling_class = VggSubsampling
        elif subsampling_name == "conv2d":
            subsampling_class = Conv2dSubsampling
        elif subsampling_name == 'residual':
            subsampling_class = ResidualSampling
        else:
            raise ValueError("subsampling must be either  'conv2d' or 'vgg' or 'residual'")

        self.conv_subsampling = subsampling_class(
            **subsampling, name=f"{name}_subsampling",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

        if positional_encoding == "sinusoid":
            self.pe = PositionalEncoding(name=f"{name}_pe")
        elif positional_encoding == "sinusoid_concat":
            self.pe = PositionalEncodingConcat(name=f"{name}_pe")
        elif positional_encoding == "subsampling":
            self.pe = tf.keras.layers.Activation("linear", name=f"{name}_pe")
        else:
            raise ValueError("positional_encoding must be either 'sinusoid' or 'subsampling'")

        self.linear = tf.keras.layers.Dense(
            dmodel, name=f"{name}_linear",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")

        self.conformer_blocks = []
        for i in range(num_blocks):
            conformer_block = ConformerBlock(
                input_dim=dmodel,
                dropout=dropout,
                fc_factor=fc_factor,
                head_size=head_size,
                num_heads=num_heads,
                mha_type=mha_type,
                kernel_size=kernel_size,
                depth_multiplier=depth_multiplier,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{name}_block_{i}"
            )
            self.conformer_blocks.append(conformer_block)

    def call(self, inputs, training=False, **kwargs):
        # input with shape [B, T, V1, V2]
        outputs = self.conv_subsampling(inputs, training=training)
        outputs = self.linear(outputs, training=training)
        pe = self.pe(outputs)
        outputs = self.do(outputs, training=training)
        for cblock in self.conformer_blocks:
            outputs = cblock([outputs, pe], training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerEncoder, self).get_config()
        conf.update(self.conv_subsampling.get_config())
        conf.update(self.linear.get_config())
        conf.update(self.do.get_config())
        conf.update(self.pe.get_config())
        for cblock in self.conformer_blocks:
            conf.update(cblock.get_config())
        return conf

class StreamingTransducerEncoder(tf.keras.Model):
    def __init__(self,
                 reductions: dict = {0: 3, 1: 2},
                 dmodel: int = 640,
                 nlayers: int = 8,
                 rnn_type: str = "lstm",
                 rnn_units: int = 2048,
                 layer_norm: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(StreamingTransducerEncoder, self).__init__(**kwargs)

        self.reshape = Reshape(name=f"{self.name}_reshape")

        self.blocks = [
            StreamingTransducerBlock(
                reduction_factor=reductions.get(i, 0),  # key is index, value is the factor
                dmodel=dmodel,
                rnn_type=rnn_type,
                rnn_units=rnn_units,
                layer_norm=layer_norm,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"{self.name}_block_{i}"
            ) for i in range(nlayers)
        ]

        self.time_reduction_factor = 1
        for i in range(nlayers):
            reduction_factor = reductions.get(i, 0)
            if reduction_factor > 0: self.time_reduction_factor *= reduction_factor

    def get_initial_state(self):
        """Get zeros states
        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, 1, P]
        """
        states = []
        for block in self.blocks:
            states.append(
                tf.stack(
                    block.rnn.get_initial_state(
                        tf.zeros([1, 1, 1], dtype=tf.float32)
                    ), axis=0
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False):
        outputs = self.reshape(inputs)
        for block in self.blocks:
            outputs = block(outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for encoder network
        Args:
            inputs (tf.Tensor): shape [1, T, F, C]
            states (tf.Tensor): shape [num_lstms, 1 or 2, 1, P]
        Returns:
            tf.Tensor: outputs with shape [1, T, E]
            tf.Tensor: new states with shape [num_lstms, 1 or 2, 1, P]
        """
        outputs = self.reshape(inputs)
        new_states = []
        for i, block in enumerate(self.blocks):
            outputs, block_states = block.recognize(outputs, states=tf.unstack(states[i], axis=0))
            new_states.append(block_states)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.reshape.get_config()
        for block in self.blocks: conf.update(block.get_config())
        return conf