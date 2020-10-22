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