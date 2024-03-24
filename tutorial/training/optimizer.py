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
learning_rate = 0.0001

# 1.2 Create the optimizer
# - The optimizer we are using is RectifiedAdam
# --- Language models are notoriously difficult to train, and large initial updates can destroy generalization ability
# --- RectifiedAdam addresses this by using the second moment to stabilize training
# - Adam is a common optimizer, but I would recommend RectifiedAdam for language models
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
# - There isn't much to do here, but do play around with the learning rate when training your model

