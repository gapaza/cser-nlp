import sys
import os
from pathlib import Path
curr_path = Path(os.path.dirname(os.path.abspath(__file__)))
root_path = curr_path.parents[1]  # parents[0] is one directory up, parents[1] is two directories up
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config
import tensorflow as tf


#        ____        _   _           _
#       / __ \      | | (_)         (_)
#      | |  | |_ __ | |_ _ _ __ ___  _ _______ _ __
#      | |  | | '_ \| __| | '_ ` _ \| |_  / _ \ '__|
#      | |__| | |_) | |_| | | | | | | |/ /  __/ |
#       \____/| .__/ \__|_|_| |_| |_|_/___\___|_|
#             | |
#             |_|


# ----------------------------------------------------
# 1. Build the optimizer
# ----------------------------------------------------

# 1.1 Define the learning rate and whether we are using JIT compilation
# - We use a relatively small learning rate for the language model due to its size
jit_compile = False
learning_rate = 0.001

# 1.2 Create the optimizer
# - The optimizer we are using is RectifiedAdam
# --- Language models are notoriously difficult to train, and high variance initial updates can destroy generalization ability
# --- RectifiedAdam addresses this by adding a rectification term to address variance in the adaptive learning rate in the initial updates
# - Adam is a common optimizer, but I would recommend RectifiedAdam for language models
# - RectifiedAdam: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/RectifiedAdam
# - Adam: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/legacy/Adam
import tensorflow_addons as tfa
optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_scheduler)





#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - For this activity, do some research on the different types of optimizers tensorflow has to offer.
# - Partial list here: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
# - Select an optimizer you find interesting, and add it above to use later

