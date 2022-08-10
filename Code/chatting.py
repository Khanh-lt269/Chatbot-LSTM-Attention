import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from Encoder import Encoder
from Decoder import Decoder
import warnings
warnings.filterwarnings("ignore")
  
# Opening JSON file
vocab_file = open('./dictionary/vocab.json')
inv_vocab_file = open('./dictionary/inv_vocab.json')
  


vocab = json.load(vocab_file)
inv_vocab = json.load(inv_vocab_file)
VOCAB_SIZE = len(vocab)


with tf.keras.utils.custom_object_scope({'Decoder': Decoder}):
    dec_model = tf.keras.models.load_model('./models/decoder_model.h5')
with tf.keras.utils.custom_object_scope({'Encoder': Encoder}):
    enc_model = tf.keras.models.load_model('./models/encoder_model.h5')

def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt
    
if __name__ == "__main__":
    
    print("##########################################")
    print("#       start chatting ver. 1.0          #")
    print("##########################################")


    prepro1 = ""
    while prepro1 != 'q':
        
        prepro1 = input("you : ")
        try:
            prepro1 = clean_text(prepro1)
            prepro = [prepro1]
            
            txt = []
            for x in prepro:
                lst = []
                for y in x.split():
                    try:
                        lst.append(vocab[y])
                    except:
                        lst.append(vocab['<OUT>'])
                txt.append(lst)
            txt = pad_sequences(txt, 13, padding='post')

            ###
            enc_op, stat = enc_model.predict( txt )

            start_index = vocab['<SOS>']
            empty_target_seq = tf.constant([[start_index]])
            stop_condition = False
            decoded_translation = ''


            while not stop_condition :

                dec_outputs , attention_weights, decoder_states_h, decoder_states_c = dec_model.predict([ empty_target_seq, enc_op ] + stat )

                sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                sampled_word = inv_vocab[str(sampled_word_index)] + ' '

                if sampled_word != '<EOS> ':
                    decoded_translation += sampled_word           

                
                if sampled_word == '<EOS> ' or len(decoded_translation.split()) > 20:
                    stop_condition = True


                empty_target_seq = tf.constant([[sampled_word_index]])
                stat = [decoder_states_h, decoder_states_c]

            print("chatbot attention : ", decoded_translation )
            print("==============================================")

        except:
            print("sorry didn't got you , please type again :( ")