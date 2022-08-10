import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Bidirectional, Concatenate
from tensorflow.keras.layers import AdditiveAttention
from AttentionLayer import AttentionLayer


class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units, len_seq, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim
        self.len_seq = len_seq

        self.embed = Embedding(self.output_vocab_size, output_dim=embedding_dim, 
            input_length=self.len_seq,
            trainable=True                  
        )
        # dec_units=400*2
        self.dec_lstm = LSTM(dec_units, return_state=True, return_sequences=True, dropout=0.05)
        self.attention = AttentionLayer(self.dec_units)

        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                    use_bias=False)
        
        self.fc = tf.keras.layers.Dense(self.output_vocab_size, activation='softmax')

    def call(self, dec_inp, enc_output, state=None):
        dec_embed = self.embed(dec_inp)
        output, h, c = self.dec_lstm(dec_embed, initial_state=state)

        context_vector, attention_weights = self.attention(
            query=output, value=enc_output
        )


        decoder_concat_input = Concatenate(axis=-1)([context_vector, output])

        attention_vector = self.Wc(decoder_concat_input)
        logits = self.fc(attention_vector)
        state = [h, c]
        return logits, attention_weights, state

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'output_vocab_size':self.output_vocab_size, 
            'embedding_dim':self.embedding_dim, 
            'dec_units':self.dec_units, 
            'len_seq':self.len_seq
        })
        return config