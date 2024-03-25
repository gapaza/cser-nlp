import sys
import os
from pathlib import Path
curr_path = Path(os.path.dirname(os.path.abspath(__file__)))
root_path = curr_path.parents[1]  # parents[0] is one directory up, parents[1] is two directories up
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

#       ______           _              _     _ _               _
#      |  ____|         | |            | |   | (_)             | |
#      | |__   _ __ ___ | |__   ___  __| | __| |_ _ __   __ _  | |     __ _ _   _  ___ _ __
#      |  __| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` | | |    / _` | | | |/ _ \ '__|
#      | |____| | | | | | |_) |  __/ (_| | (_| | | | | | (_| | | |___| (_| | |_| |  __/ |
#      |______|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, | |______\__,_|\__, |\___|_|
#                                                        __/ |               __/ |
#                                                       |___/               |___/


# ---------------------------------------------------------------------
# 1. Define the embedding layer for the language model
# ---------------------------------------------------------------------
# - For this, we will be using the keras_nlp TokenAndPositionEmbedding class
# https://keras.io/api/keras_nlp/modeling_layers/token_and_position_embedding
from keras_nlp.layers import TokenAndPositionEmbedding

# 1.1 Define the embedding dimension and the maximum sequence length
# - We already defined the max sequence length when creating the vocabulary
# - We also need the vocabulary size to create the embedding layer
from tutorial.data_preparation.build_vocabulary import max_seq_len, vocab_size
max_seq_len = max_seq_len
vocab_size = vocab_size
embed_dim = 256

# 1.2 Create the TokenAndPositionEmbedding layer
# - NOTE: we mask zero tokens to ensure that the model does not consider padding tokens during training
#         this saves computational resources and speeds up training
embedding_layer = TokenAndPositionEmbedding(
    vocab_size,
    max_seq_len,
    embed_dim,
    mask_zero=True
)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try using the embedding layer with the following
# 1. Create a python string that is a requirement
# 2. Encode the requirement using the tokenizer from tokenize.py
# 3. Pass the encoded requirement through the embedding layer
# - You can call the embedding layer by simply passing the encoded requirement as an argument
# --- e.g. embedding_layer(encoded_requirement)
# 4. Print the output, and observe the shape of the output
# - What are the dimensions of the output? Why are they such?
# YOUR EXPERIMENTAL CODE HERE
from tutorial.data_preparation.build_tokenizer import encode_tf


