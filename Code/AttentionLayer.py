import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate
from tensorflow.keras.layers import AdditiveAttention


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # units=400*2
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attn_layer = AdditiveAttention(use_scale=True) 

    def call(self, query, value):
        w1_query = self.W1(query)
        w2_key = self.W2(value)
        context_vector, attention_weights = self.attn_layer(
            [w1_query, value, w2_key],
            return_attention_scores = True
        )

        return context_vector, attention_weights
