import typing
import tensorflow as tf
from tamil_tech.utils.tf_utils import *

L2 = tf.keras.regularizers.l2(1e-6)

class Concatenation(tf.keras.layers.Layer):
    def __init__(self, dmodel, **kwargs):
        super(Concatenation, self).__init__(**kwargs)
        self.concat = tf.keras.layers.Concatenate(axis=-1, name=f"{self.name}_concat")
        self.joint = tf.keras.layers.Dense(dmodel, name=f"{self.name}_joint")

    def call(self, inputs, training=False):
        outputs = self.concat(inputs)
        return self.joint(outputs, training=training)

    def get_config(self):
        conf = super(Concatenation).get_config()
        conf.update(self.concat.get_config())
        conf.update(self.joint.get_config())
        return conf

class GLU(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 name="glu_activation",
                 **kwargs):
        super(GLU, self).__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        a, b = tf.split(inputs, 2, axis=self.axis)
        b = tf.nn.sigmoid(b)
        return tf.multiply(a, b)

    def get_config(self):
        conf = super(GLU, self).get_config()
        conf.update({"axis": self.axis})
        return conf

class Embedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 contraint=None,
                 regularizer=None,
                 initializer=None,
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.contraint = tf.keras.constraints.get(contraint)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings", dtype=tf.float32,
            shape=[self.vocab_size, self.embed_dim],
            initializer=self.initializer,
            trainable=True, regularizer=self.regularizer,
            constraint=self.contraint
        )
        self.built = True

    def call(self, inputs):
        outputs = tf.cast(tf.expand_dims(inputs, axis=-1), dtype=tf.int32)
        return tf.gather_nd(self.embeddings, outputs)

    def get_config(self):
        conf = super(Embedding, self).get_config()
        conf.update({
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "contraint": self.contraint,
            "regularizer": self.regularizer,
            "initializer": self.initializer
        })
        return conf

class TimeReduction(tf.keras.layers.Layer):
    def __init__(self, factor: int, name: str = "time_reduction", **kwargs):
        super(TimeReduction, self).__init__(name=name, **kwargs)
        self.factor = factor

    def build(self, input_shape):
        batch_size = input_shape[0]
        feat_dim = input_shape[-1]
        self.reshape = tf.keras.layers.Reshape([batch_size, -1, feat_dim * self.factor])

    def call(self, inputs, **kwargs):
        return self.reshape(inputs)

    def get_config(self):
        config = super(TimeReduction, self).get_config()
        config.update({"factor": self.factor})
        return config

    def from_config(self, config):
        return self(**config)

class PositionalEncoding(tf.keras.layers.Layer):
    def build(self, input_shape):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(max_len, dmodel):
        pos = tf.expand_dims(tf.range(max_len - 1, -1, -1.0, dtype=tf.float32), axis=1)
        index = tf.expand_dims(tf.range(0, dmodel, dtype=tf.float32), axis=0)

        pe = pos * (1 / tf.pow(10000.0, (2 * (index // 2)) / dmodel))

        # Sin cos will be [max_len, size // 2]
        # we add 0 between numbers by using padding and reshape
        sin = tf.pad(tf.expand_dims(tf.sin(pe[:, 0::2]), -1),
                     [[0, 0], [0, 0], [0, 1]], mode="CONSTANT", constant_values=0)
        sin = tf.reshape(sin, [max_len, dmodel])
        cos = tf.pad(tf.expand_dims(tf.cos(pe[:, 1::2]), -1),
                     [[0, 0], [0, 0], [1, 0]], mode="CONSTANT", constant_values=0)
        cos = tf.reshape(cos, [max_len, dmodel])
        # Then add sin and cos, which results in [time, size]
        pe = tf.add(sin, cos)
        return tf.expand_dims(pe, axis=0)  # [1, time, size]

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)

    def get_config(self):
        conf = super(PositionalEncoding, self).get_config()
        return conf

class PositionalEncodingConcat(tf.keras.layers.Layer):
    def build(self, input_shape):
        dmodel = input_shape[-1]
        assert dmodel % 2 == 0, f"Input last dim must be even: {dmodel}"

    @staticmethod
    def encode(max_len, dmodel):
        pos = tf.range(max_len - 1, -1, -1.0, dtype=tf.float32)

        index = tf.range(0, dmodel, 2.0, dtype=tf.float32)
        index = 1 / tf.pow(10000.0, (index / dmodel))

        sinusoid = tf.einsum("i,j->ij", pos, index)
        pos = tf.concat([tf.sin(sinusoid), tf.cos(sinusoid)], axis=-1)

        return tf.expand_dims(pos, axis=0)

    def call(self, inputs, **kwargs):
        # inputs shape [B, T, V]
        _, max_len, dmodel = shape_list(inputs)
        pe = self.encode(max_len, dmodel)
        return tf.cast(pe, dtype=inputs.dtype)

    def get_config(self):
        conf = super(PositionalEncoding, self).get_config()
        return conf

class VggSubsampling(tf.keras.layers.Layer):
    def __init__(self,
                 filters: tuple or list = (32, 64),
                 kernel_size: int or list or tuple = 3,
                 strides: int or list or tuple = 2,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="VggSubsampling",
                 **kwargs):
        super(VggSubsampling, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0], kernel_size=kernel_size, strides=1,
            padding="same", name=f"{name}_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters[0], kernel_size=kernel_size, strides=1,
            padding="same", name=f"{name}_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(
            pool_size=strides,
            padding="same", name=f"{name}_maxpool_1"
        )
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters[1], kernel_size=kernel_size, strides=1,
            padding="same", name=f"{name}_conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=filters[1], kernel_size=kernel_size, strides=1,
            padding="same", name=f"{name}_conv_4",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(
            pool_size=strides,
            padding="same", name=f"{name}_maxpool_2"
        )
        self.time_reduction_factor = self.maxpool1.pool_size[0] + self.maxpool2.pool_size[0]

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv3(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv4(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool2(outputs, training=training)

        return merge_two_last_dims(outputs)

    def get_config(self):
        conf = super(VggSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.maxpool1.get_config())
        conf.update(self.conv3.get_config())
        conf.update(self.conv4.get_config())
        conf.update(self.maxpool2.get_config())
        return conf

class Conv2dSubsampling(tf.keras.layers.Layer):
    def __init__(self,
                 filters: int,
                 strides: list or tuple or int = 2,
                 kernel_size: int or list or tuple = 3,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="Conv2dSubsampling",
                 **kwargs):
        super(Conv2dSubsampling, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding="same", name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, padding="same", name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.time_reduction_factor = self.conv1.strides[0] + self.conv2.strides[0]

    def call(self, inputs, training=False, **kwargs):
        outputs = self.conv1(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        return merge_two_last_dims(outputs)

    def get_config(self):
        conf = super(Conv2dSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        return conf

class MHSA(tf.keras.layers.Layer):
    def __init__(self,
                 num_heads,
                 head_size,
                 output_size: int = None,
                 dropout: float = 0.0,
                 use_projection_bias: bool = True,
                 return_attn_coef: bool = False,
                 kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
                 kernel_regularizer: typing.Union[str, typing.Callable] = None,
                 kernel_constraint: typing.Union[str, typing.Callable] = None,
                 bias_initializer: typing.Union[str, typing.Callable] = "zeros",
                 bias_regularizer: typing.Union[str, typing.Callable] = None,
                 bias_constraint: typing.Union[str, typing.Callable] = None,
                 **kwargs):
        super(MHSA, self).__init__(**kwargs)

        if output_size is not None and output_size < 1:
            raise ValueError("output_size must be a positive number")

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.dropout = tf.keras.layers.Dropout(dropout, name="dropout")
        self._droput_rate = dropout

    def build(self, input_shape):
        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else num_key_features
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )
        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

    def call_qkv(self, query, key, value, training=False):
        # verify shapes
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                "the number of elements in 'key' must be equal to "
                "the same as the number of elements in 'value'"
            )
        # Linear transformations
        query = tf.einsum("...NI,HIO->...NHO", query, self.query_kernel)
        key = tf.einsum("...MI,HIO->...MHO", key, self.key_kernel)
        value = tf.einsum("...MI,HIO->...MHO", value, self.value_kernel)

        return query, key, value

    def call_attention(self, query, key, value, logits, training=False, mask=None):
        if mask is not None:
            if len(mask.shape) < 2:
                raise ValueError("'mask' must have atleast 2 dimensions")
            if query.shape[-2] != mask.shape[-2]:
                raise ValueError(
                    "mask's second to last dimension must be equal to "
                    "the number of elements in 'query'"
                )
            if key.shape[-2] != mask.shape[-1]:
                raise ValueError(
                    "mask's last dimension must be equal to the number of elements in 'key'"
                )
        # apply mask
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(mask.shape) != len(logits.shape):
                mask = tf.expand_dims(mask, -3)

            logits += -10e9 * (1.0 - mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum("...NHI,HIO->...NO", multihead_output, self.projection_kernel)

        if self.projection_bias is not None:
            output += self.projection_bias

        return output, attn_coef

    def call(self, inputs, training=False, mask=None):
        query, key, value = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        output, attn_coef = self.call_attention(query, key, value, logits,
                                                training=training, mask=mask)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = (
            input_shape[2][-1] if len(input_shape) > 2 else input_shape[1][-1]
        )
        output_size = (
            self.output_size if self.output_size is not None else num_value_features
        )

        output_shape = input_shape[0][:-1] + (output_size,)

        if self.return_attn_coef:
            num_query_elements = input_shape[0][-2]
            num_key_elements = input_shape[1][-2]
            attn_coef_shape = input_shape[0][:-2] + (
                self.num_heads,
                num_query_elements,
                num_key_elements,
            )

            return output_shape, attn_coef_shape
        else:
            return output_shape

    def get_config(self):
        config = super().get_config()

        config.update(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout=self._droput_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
        )

        return config

class RelativePositionMHSA(MHSA):
    def build(self, input_shape):
        num_pos_features = input_shape[-1][-1]
        self.pos_kernel = self.add_weight(
            name="pos_kernel",
            shape=[self.num_heads, num_pos_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        self.pos_bias_u = self.add_weight(
            name="pos_bias_u",
            shape=[self.num_heads, self.head_size],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint
        )
        self.pos_bias_v = self.add_weight(
            name="pos_bias_v",
            shape=[self.num_heads, self.head_size],
            regularizer=self.kernel_regularizer,
            initializer=self.kernel_initializer,
            constraint=self.kernel_constraint
        )
        super(RelativePositionMHSA, self).build(input_shape[:-1])

    @staticmethod
    def relative_shift(x):
        x_shape = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, [x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2]])
        x = tf.reshape(x[:, :, 1:, :], x_shape)
        return x

    def call(self, inputs, training=False, mask=None):
        query, key, value, pos = inputs

        query, key, value = self.call_qkv(query, key, value, training=training)

        pos = tf.einsum("...MI,HIO->...MHO", pos, self.pos_kernel)

        query_with_u = query + self.pos_bias_u
        query_with_v = query + self.pos_bias_v

        logits_with_u = tf.einsum("...NHO,...MHO->...HNM", query_with_u, key)
        logits_with_v = tf.einsum("...NHO,...MHO->...HNM", query_with_v, pos)
        logits_with_v = self.relative_shift(logits_with_v)

        logits = logits_with_u + logits_with_v

        depth = tf.constant(self.head_size, dtype=tf.float32)
        logits /= tf.sqrt(depth)

        output, attn_coef = self.call_attention(query, key, value, logits,
                                                training=training, mask=mask)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

class FeedForwardModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="ff_module",
                 **kwargs):
        super(FeedForwardModule, self).__init__(name=name, **kwargs)
        self.fc_factor = fc_factor
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.ffn1 = tf.keras.layers.Dense(
            4 * input_dim, name=f"{name}_dense_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name=f"{name}_swish_activation")
        self.do1 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_1")
        self.ffn2 = tf.keras.layers.Dense(
            input_dim, name=f"{name}_dense_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do2 = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout_2")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.ffn1(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, self.fc_factor * outputs])
        return outputs

    def get_config(self):
        conf = super(FeedForwardModule, self).get_config()
        conf.update({"fc_factor": self.fc_factor})
        conf.update(self.ln.get_config())
        conf.update(self.ffn1.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.res_add.get_config())
        return conf

class PointWiseFeedForwardModule(tf.keras.layers.Layer):
    def __init__(self,
                 size,
                 output_size,
                 activation="relu",
                 dropout=0.1,
                 name="point_wise_ffn",
                 **kwargs):
        super(PointWiseFeedForwardModule, self).__init__(name=name, **kwargs)
        self.ffn1 = tf.keras.layers.Dense(units=size, activation=activation)
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.ffn2 = tf.keras.layers.Dense(units=output_size)
        self.do2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ffn1(inputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.ffn2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(PointWiseFeedForwardModule, self).get_config()
        conf.update(self.ffn1.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.ffn2.get_config())
        conf.update(self.do2.get_config())
        return conf

class MHSAModule(tf.keras.layers.Layer):
    def __init__(self,
                 head_size,
                 num_heads,
                 dropout=0.0,
                 mha_type="relmha",
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="mhsa_module",
                 **kwargs):
        super(MHSAModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        if mha_type == "relmha":
            self.mha = RelativePositionMHSA(
                name=f"{name}_mhsa",
                head_size=head_size, num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
        elif mha_type == "mha":
            self.mha = MHSA(
                name=f"{name}_mhsa",
                head_size=head_size, num_heads=num_heads,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
        else:
            raise ValueError("mha_type must be either 'mha' or 'relmha'")
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")
        self.mha_type = mha_type

    def call(self, inputs, training=False, **kwargs):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.ln(inputs, training=training)
        if self.mha_type == "relmha":
            outputs = self.mha([outputs, outputs, outputs, pos], training=training)
        else:
            outputs = outputs + pos
            outputs = self.mha([outputs, outputs, outputs], training=training)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(MHSAModule, self).get_config()
        conf.update({"mha_type": self.mha_type})
        conf.update(self.ln.get_config())
        conf.update(self.mha.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf

class ConvModule(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 kernel_size=32,
                 dropout=0.0,
                 depth_multiplier=1,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conv_module",
                 **kwargs):
        super(ConvModule, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.pw_conv_1 = tf.keras.layers.Conv2D(
            filters=2 * input_dim, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.glu = GLU(name=f"{name}_glu")
        self.dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, 1), strides=1,
            padding="same", name=f"{name}_dw_conv",
            depth_multiplier=depth_multiplier,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.bn = tf.keras.layers.BatchNormalization(
            name=f"{name}_bn",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer
        )
        self.swish = tf.keras.layers.Activation(
            tf.keras.activations.swish, name=f"{name}_swish_activation")
        self.pw_conv_2 = tf.keras.layers.Conv2D(
            filters=input_dim, kernel_size=1, strides=1,
            padding="valid", name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.do = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")
        self.res_add = tf.keras.layers.Add(name=f"{name}_add")

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = tf.expand_dims(outputs, axis=2)
        outputs = self.pw_conv_1(outputs, training=training)
        outputs = self.glu(outputs)
        outputs = self.dw_conv(outputs, training=training)
        outputs = self.bn(outputs, training=training)
        outputs = self.swish(outputs)
        outputs = self.pw_conv_2(outputs, training=training)
        outputs = tf.squeeze(outputs, axis=2)
        outputs = self.do(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(ConvModule, self).get_config()
        conf.update(self.ln.get_config())
        conf.update(self.pw_conv_1.get_config())
        conf.update(self.glu.get_config())
        conf.update(self.dw_conv.get_config())
        conf.update(self.bn.get_config())
        conf.update(self.swish.get_config())
        conf.update(self.pw_conv_2.get_config())
        conf.update(self.do.get_config())
        conf.update(self.res_add.get_config())
        return conf

class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_dim,
                 dropout=0.0,
                 fc_factor=0.5,
                 head_size=36,
                 num_heads=4,
                 mha_type="relmha",
                 kernel_size=32,
                 depth_multiplier=1,
                 kernel_regularizer=L2,
                 bias_regularizer=L2,
                 name="conformer_block",
                 **kwargs):
        super(ConformerBlock, self).__init__(name=name, **kwargs)
        self.ffm1 = FeedForwardModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor, name=f"{name}_ff_module_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.mhsam = MHSAModule(
            mha_type=mha_type,
            head_size=head_size, num_heads=num_heads,
            dropout=dropout, name=f"{name}_mhsa_module",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.convm = ConvModule(
            input_dim=input_dim, kernel_size=kernel_size,
            dropout=dropout, name=f"{name}_conv_module",
            depth_multiplier=depth_multiplier,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ffm2 = FeedForwardModule(
            input_dim=input_dim, dropout=dropout,
            fc_factor=fc_factor, name=f"{name}_ff_module_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )
        self.ln = tf.keras.layers.LayerNormalization(
            name=f"{name}_ln",
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=kernel_regularizer
        )

    def call(self, inputs, training=False, **kwargs):
        inputs, pos = inputs  # pos is positional encoding
        outputs = self.ffm1(inputs, training=training)
        outputs = self.mhsam([outputs, pos], training=training)
        outputs = self.convm(outputs, training=training)
        outputs = self.ffm2(outputs, training=training)
        outputs = self.ln(outputs, training=training)
        return outputs

    def get_config(self):
        conf = super(ConformerBlock, self).get_config()
        conf.update(self.ffm1.get_config())
        conf.update(self.mhsam.get_config())
        conf.update(self.convm.get_config())
        conf.update(self.ffm2.get_config())
        conf.update(self.ln.get_config())
        return conf