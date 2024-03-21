#
#      __      __                 _             _
#      \ \    / /                | |           | |
#       \ \  / /___    ___  __ _ | |__   _   _ | |  __ _  _ __  _   _
#        \ \/ // _ \  / __|/ _` || '_ \ | | | || | / _` || '__|| | | |
#         \  /| (_) || (__| (_| || |_) || |_| || || (_| || |   | |_| |
#          \/  \___/  \___|\__,_||_.__/  \__,_||_| \__,_||_|    \__, |
#                                                                __/ |
#                                                               |___/
#

import config
import tensorflow as tf
from keras.layers import TextVectorization
import re
import os




# --- Requirement Data ---
req_data = config.req_data


# --- Vectorization ---
max_seq_len = 50
max_tokens = 20000
tokenizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_seq_len,
)
tokenizer.adapt(req_data)
vocabulary = tokenizer.get_vocabulary()
vocabulary.extend(["[start]", "[end]"])
tokenizer.set_vocabulary(vocabulary)
vocabulary = tokenizer.get_vocabulary()
vocab_size = len(vocabulary)
print('Vocabulary Size: ', vocab_size)
print('Vocabulary: ', vocabulary)


# --- Specific Tokens ---
id2token = dict(enumerate(vocabulary))
token2id = {y: x for x, y in id2token.items()}
start_token_label = '[start]'
end_token_label = '[end]'
start_token_id = tokenizer(['[start]']).numpy()[0][0]
end_token_id = tokenizer(['[end]']).numpy()[0][0]


# --- Encoding Functions ---

def encode(input):
    encoded_input = tokenizer(input)
    return encoded_input.numpy()

@tf.function
def encode_tf(input):
    encoded_input = tokenizer(input)
    return encoded_input

















# if __name__ == '__main__':
#     run()



