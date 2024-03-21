import tensorflow as tf
import config
import os
from keras.utils import plot_model
import model.vocabulary as vocabulary

from model.RequirementDecoder import RequirementDecoder

def get_requirement_decoder(checkpoint_path=None):

    # 1. Create model
    model = RequirementDecoder()

    # 2. Implicit build
    zeroed_input = tf.zeros((1, vocabulary.max_seq_len))
    model(zeroed_input)

    # 3. Load Weights
    if checkpoint_path:
        model.load_weights(checkpoint_path).expect_partial()

    # 4. Summary
    model.summary()

    return model

























