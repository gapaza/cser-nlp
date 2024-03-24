import config

#       _______    _              _          _   _
#      |__   __|  | |            (_)        | | (_)
#         | | ___ | | _____ _ __  _ ______ _| |_ _  ___  _ __
#         | |/ _ \| |/ / _ \ '_ \| |_  / _` | __| |/ _ \| '_ \
#         | | (_) |   <  __/ | | | |/ / (_| | |_| | (_) | | | |
#         |_|\___/|_|\_\___|_| |_|_/___\__,_|\__|_|\___/|_| |_|
print_output = False


# -------------------------------------------------------------
# 1. Define encoding functions to simplify dataset creation
# -------------------------------------------------------------
from tutorial.data_preparation.build_vocabulary import tokenizer, id2token
import tensorflow as tf

# 1.1. Encode a list of requirements into a list of token ids
# - NOTE: this function takes a list of requirement sentences and returns a list of token id sequences
def encode(input):
    encoded_input = tokenizer(input)
    return encoded_input.numpy()

# 1.2. Encode a list of requirements into a list of token ids using a TensorFlow function
# - NOTE: this function is a TensorFlow wrapper around the encode function
#         the purpose of this wrapper is to speed up the encoding process
@tf.function
def encode_tf(input):
    encoded_input = tokenizer(input)
    return encoded_input

#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try writing your own requirements in python strings and encoding them using the encode function
# --- Ensure to abide by the input format of the encode function!
# - Try decoding them back to a list of tokens using the id2token dictionary defined previously
# - Try encoding a sample requirement with a word that is unlikely to be found in the vocabulary...
# --- What is the token for the word that is not found?
# YOUR EXPERIMENTAL CODE HERE



