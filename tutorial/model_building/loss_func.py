import config

#       _                     ______                _   _
#      | |                   |  ____|              | | (_)
#      | |     ___  ___ ___  | |__ _   _ _ __   ___| |_ _  ___  _ __  ___
#      | |    / _ \/ __/ __| |  __| | | | '_ \ / __| __| |/ _ \| '_ \/ __|
#      | |___| (_) \__ \__ \ | |  | |_| | | | | (__| |_| | (_) | | | \__ \
#      |______\___/|___/___/ |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/
# - Now that we have all the components for our model, we can formulate the loss function


# -----------------------------------------------
# 1. Define the loss and metrics
# -----------------------------------------------
# - For the loss function, we will be using the SparseCategoricalCrossentropy loss
# - This loss function is used for multi-class classification problems
# --- In our case, we are classifying each token in the vocabulary
import tensorflow as tf
import keras_nlp


# 1.1 Define the loss function
# https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# 1.2 Define the perplexity metric
# - This is a common metric used in language modeling
# - It measures how confident the model is in its predictions
# - We don't want the perplexity metric to be influenced by padding tokens
# https://keras.io/api/keras_nlp/metrics/perplexity/
perplexity_tracker = keras_nlp.metrics.Perplexity(mask_token_id=0)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try using the loss function with the layers developed thus far
# - To facilitate things, here is a input and label pair to start with
# 1. Encode the input and label
# 2. Pass it through the layers developed
# 3. Calculate the loss using the loss function
# - You can call the loss function by passing the output of the output layer and the encoded label as arguments
# 4. Calculate the perplexity using the perplexity tracker
# - You can call the perplexity tracker by passing the output of the output layer and the encoded label as arguments
from tutorial.data_preparation.build_vocabulary import start_token_label, end_token_label
from tutorial.data_preparation.tokenize import encode_tf
from tutorial.model_building.embedding_layer import embedding_layer
from tutorial.model_building.decoder_layer import decoder_layer
from tutorial.model_building.output_layer import output_layer, activation_layer
input_text = start_token_label + ' ' + 'The ADCS shall'
label_text = 'The ADCS shall' + ' ' + end_token_label



