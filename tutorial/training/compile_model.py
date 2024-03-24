import config
import os

#        _____                      _ _        __  __           _      _
#       / ____|                    (_) |      |  \/  |         | |    | |
#      | |     ___  _ __ ___  _ __  _| | ___  | \  / | ___   __| | ___| |
#      | |    / _ \| '_ ` _ \| '_ \| | |/ _ \ | |\/| |/ _ \ / _` |/ _ \ |
#      | |___| (_) | | | | | | |_) | | |  __/ | |  | | (_) | (_| |  __/ |
#       \_____\___/|_| |_| |_| .__/|_|_|\___| |_|  |_|\___/ \__,_|\___|_|
#                            | |
#                            |_|


# --------------------------------------------------------
# 1. Build an instance of our RequirementsDecoder
# --------------------------------------------------------

from tutorial.model_building.RequirementDecoder import RequirementDecoder
model = RequirementDecoder()




# --------------------------------------------------------
# 2. Build the model
# --------------------------------------------------------
# - Tensorflow models can be built a couple different ways
# - We will take an "implicit" approach to building our model
# - This is done by simply conducting a forward pass with any random input

# 2.1 Create random input
# - The input tensor needs to have an appropriate shape
# --- This shape is defined by the batch size and sequence length
# --- NOTE: the batch size can be any value, it doesn't have to match the batch size of our dataset
# - For our input, we will use a tensor of zeros
import tensorflow as tf
from tutorial.data_preparation.build_vocabulary import max_seq_len
max_seq_len = max_seq_len
build_input = tf.zeros((1, max_seq_len))

# 2.2 Implicit build
model(build_input)




# --------------------------------------------------------
# 3. Load weights (optional)
# --------------------------------------------------------
# - If you have a pre-trained model, you can load its weights into the model you just built
# - This is useful if you are fine-tuning a model, or just continuing training at a later time
# - This will be covered in more detail when conducting inference on the model

# 3.1 Define load path
load_path = None


# 3.2 Load the model
# - The expect partial function silences any warning output in case the loaded weights are a subset of the model's weights
if load_path:
    model.load_weights(load_path).expect_partial()




#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - To output information about your model, you can call the summary() function on the model object
# - Play around with the model's architecture to see how it changes the number of parameters
# --- Number of attention heads, feed forward dimensions, etc.
# YOUR CODE HERE





