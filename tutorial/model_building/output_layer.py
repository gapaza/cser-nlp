import sys
import os
from pathlib import Path
curr_path = Path(os.path.dirname(os.path.abspath(__file__)))
root_path = curr_path.parents[1]  # parents[0] is one directory up, parents[1] is two directories up
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

#        ____        _               _     _
#       / __ \      | |             | |   | |
#      | |  | |_   _| |_ _ __  _   _| |_  | |     __ _ _   _  ___ _ __
#      | |  | | | | | __| '_ \| | | | __| | |    / _` | | | |/ _ \ '__|
#      | |__| | |_| | |_| |_) | |_| | |_  | |___| (_| | |_| |  __/ |
#       \____/ \__,_|\__| .__/ \__,_|\__| |______\__,_|\__, |\___|_|
#                       | |                             __/ |
#                       |_|                            |___/
# - The purpose of the output layer is to predict the next token in the sequence
# - NOTE: the output layer has an output for each sequence element

# ---------------------------------------------------------------------
# 1. Define the output layer for the language model
# ---------------------------------------------------------------------
# - For this, we will be using a simple dense layer with a softmax activation
# https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
from keras import layers

# 1.1 Define the output size of the layer
# - There should be as many units (outputs) as there are tokens in the vocabulary
# --- This has already been defined in the build_vocabulary.py script
from tutorial.data_preparation.build_vocabulary import vocab_size
output_size = vocab_size

# 1.2 Create the dense output layer with a softmax activation layer
# - The output of this is a probability distribution over the vocabulary for each sequence element
output_layer = layers.Dense(
    output_size,
    name='output_layer'
)
activation_layer = layers.Activation(
    'softmax',
    name='softmax_activation'
)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try using the output layer with the following
# 1. Create a python string that is a requirement
# 2. Encode the requirement using the tokenizer from tokenize.py
# 3. Pass the encoded requirement through the embedding layer from embedding_layer.py
# 4. Pass the output of the embedding layer through the decoder layer
# 5. Pass the output of the decoder layer through the output layer (both the dense and activation layers)
# - You can call the output layer by simply passing the output of the decoder layer as an argument
# 6. Print the output, and observe the shape of the output
# - Try to sample a specific sequence element from the output, and observe the probability distribution
# - You can also try to select index of the most probable token using the argmax function
# - If you can do this, you can use the id2token dictionary from build_vocabulary.py to get the predicted token!
# YOUR EXPERIMENTAL CODE HERE
from tutorial.data_preparation.build_vocabulary import id2token
from tutorial.data_preparation.build_tokenizer import encode_tf
from tutorial.model_building.embedding_layer import embedding_layer
from tutorial.model_building.decoder_layer import decoder_layer
