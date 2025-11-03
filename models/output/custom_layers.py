"""
Custom Keras Layers for Model Loading

This file contains custom layer definitions needed for loading models with
custom architecture components. It's designed to be lightweight and have
no external dependencies beyond tensorflow/keras.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Dropout, LayerNormalization, MultiHeadAttention
)


class TransformerBlock(tf.keras.layers.Layer):
    """Single Transformer block with multi-head attention and feed-forward network"""

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(d_model),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout_rate,
        })
        return config


class CNNLSTM(tf.keras.layers.Layer):
    """CNN+LSTM hybrid layer combining convolutional and recurrent processing"""

    def __init__(self, filters=64, kernel_size=3, lstm_units=128, **kwargs):
        super(CNNLSTM, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units

    def build(self, input_shape):
        # Layer definitions would go here
        pass

    def call(self, inputs):
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'lstm_units': self.lstm_units,
        })
        return config
