import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate
from tensorflow.keras.layers import AdditiveAttention


class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units, len_seq,**kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.len_seq = len_seq
        self.embedding_dim = embedding_dim
        
        self.embed = Embedding(self.input_vocab_size, output_dim=self.embedding_dim, 
                  input_length=self.len_seq,
                  trainable=True                  
                  )
        self.enc_lstm = Bidirectional(LSTM(self.enc_units, return_state=True, dropout=0.05, return_sequences = True))

    def call(self, enc_inp, state=None):
        enc_embed = self.embed(enc_inp)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = self.enc_lstm(enc_embed)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])

        enc_states = [state_h, state_c]

        return encoder_outputs, enc_states

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_vocab_size': self.input_vocab_size, 
            'embedding_dim': self.embedding_dim, 
            'enc_units': self.enc_units, 
            'len_seq': self.len_seq
        })
        return config

        
