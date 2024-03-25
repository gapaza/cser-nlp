import sys
import os
from pathlib import Path
curr_path = Path(os.path.dirname(os.path.abspath(__file__)))
root_path = curr_path.parents[1]  # parents[0] is one directory up, parents[1] is two directories up
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

#       _____                     _             _
#      |  __ \                   | |           | |
#      | |  | | ___  ___ ___   __| | ___ _ __  | |     __ _ _   _  ___ _ __
#      | |  | |/ _ \/ __/ _ \ / _` |/ _ \ '__| | |    / _` | | | |/ _ \ '__|
#      | |__| |  __/ (_| (_) | (_| |  __/ |    | |___| (_| | |_| |  __/ |
#      |_____/ \___|\___\___/ \__,_|\___|_|    |______\__,_|\__, |\___|_|
#                                                            __/ |
#                                                           |___/

# ---------------------------------------------------------------------
# 1. Define the embedding layer for the language model
# ---------------------------------------------------------------------
# - For this, we will be using the keras_nlp TransformerDecoder class
# https://keras.io/api/keras_nlp/modeling_layers/transformer_decoder/
from keras_nlp.layers import TransformerDecoder


# 1.1 Define the dimension of the decoder feed forward networks and the number of attention heads
# - We can also define if we want to use dropout in the decoder
dropout_rate = 0.2
ff_dim = 256
num_heads = 2  # We only use 2 attention heads for the small LLM

# 1.2 Create the TransformerDecoder layer
decoder_layer = TransformerDecoder(
    ff_dim,
    num_heads,
    dropout=dropout_rate,
    name='transformer_decoder'
)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try using the decoder layer with the following
# 1. Create a python string that is a requirement
# 2. Encode the requirement using the tokenizer from tokenize.py
# 3. Pass the encoded requirement through the embedding layer from embedding_layer.py
# 4. Pass the output of the embedding layer through the decoder layer
# - You can call the decoder layer by simply passing the output of the embedding layer as an argument
# 5. Print the output, and observe the shape of the output
# - What are the dimensions of the output? Why are they such?
# - Are the output values of the decoder different from the input values?
# YOUR EXPERIMENTAL CODE HERE
from tutorial.data_preparation.build_tokenizer import encode_tf
from tutorial.model_building.embedding_layer import embedding_layer



