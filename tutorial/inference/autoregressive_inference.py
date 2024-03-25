import sys
import os
from pathlib import Path
curr_path = Path(os.path.dirname(os.path.abspath(__file__)))
root_path = curr_path.parents[1]  # parents[0] is one directory up, parents[1] is two directories up
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

#                     _                                         _             _____        __
#          /\        | |                                       (_)           |_   _|      / _|
#         /  \  _   _| |_ ___  _ __ ___  __ _ _ __ ___  ___ ___ ___   _____    | |  _ __ | |_ ___ _ __ ___ _ __   ___ ___
#        / /\ \| | | | __/ _ \| '__/ _ \/ _` | '__/ _ \/ __/ __| \ \ / / _ \   | | | '_ \|  _/ _ \ '__/ _ \ '_ \ / __/ _ \
#       / ____ \ |_| | || (_) | | |  __/ (_| | | |  __/\__ \__ \ |\ V /  __/  _| |_| | | | ||  __/ | |  __/ | | | (_|  __/
#      /_/    \_\__,_|\__\___/|_|  \___|\__, |_|  \___||___/___/_| \_/ \___| |_____|_| |_|_| \___|_|  \___|_| |_|\___\___|
#                                        __/ |
#                                       |___/
# - Now, we will generate new requirements from our trained model!


# -----------------------------------------------------
# 1. Load the trained model
# -----------------------------------------------------

# 1.1 Create a new model instance
from tutorial.model_building.RequirementDecoder import RequirementDecoder
model = RequirementDecoder()

# 1.2 Get the path of the model to load
load_model_name = 'tutorial_model'
load_model_path = os.path.join(config.trained_models_dir, load_model_name)

# 1.3 Load the model weights
model.load_weights(load_model_path).expect_partial()


# -----------------------------------------------------
# 2. Autoregressively sample the model
# -----------------------------------------------------
# - For this step, we will sample the model in a loop, appending the generated tokens to the input sequence
import tensorflow as tf

# 2.1 Define an initial input sequence only containing the start token
# - We already have the start token from the vocabulary
from tutorial.data_preparation.build_vocabulary import start_token_id, end_token_id, id2token
input_tokens = [start_token_id]

# 2.2 Define the list that will store the generated tokens
output_tokens = []

# 2.3 Now loop through the model and generate new tokens until either:
# - 1) The maximum sequence length is reached
# - 2) The end token is generated
from tutorial.data_preparation.build_vocabulary import max_seq_len
for _ in range(max_seq_len):

    # 1. Create the input tensor
    input_tensor = tf.convert_to_tensor(input_tokens)

    # 2. We need the input to have a batch dimension
    input_tensor = tf.expand_dims(input_tensor, axis=0)

    # 3. Pass through the model
    decoder_output = model(input_tensor, training=False)

    # 4. Now, determine which token is the "next one", as the decoder generates a token for each sequence element
    next_token_idx = len(input_tokens) - 1
    next_token_prob_dist = decoder_output[:, next_token_idx, :]

    # 5. Next, we need to sample from the probability distribution of next tokens
    # - We will use the tensorflow categorical function to sample from the distribution
    # - To use this function, we need to convert the probabilities to log probabilities
    # - NOTE: we add a small value to prevent numerical instability issues
    next_token_log_probs = tf.math.log(next_token_prob_dist + 1e-10)
    samples = tf.random.categorical(next_token_log_probs, 1)  # shape (batch, 1)
    next_token_id = tf.squeeze(samples, axis=-1)
    next_token_id = next_token_id.numpy()[0]

    # 6. Append the next token to the input sequence
    input_tokens.append(next_token_id)

    # 7. If the next token is the end token, we break the loop
    if next_token_id == end_token_id:
        break

    # 8. Finally, we convert the generated token id to the corresponding token and append it to the output list
    next_token = id2token[next_token_id]
    output_tokens.append(next_token)


# -----------------------------------------------------
# 3. Print the generated requirement
# -----------------------------------------------------
# - All that is left to do is print the requirement that was generated!
print('Generated Requirement:', ' '.join(output_tokens))







#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - This repo comes with a variety of pre-trained model, trained to generate requirements for different subsystems
# - Try loading a different model and generating requirements from it!































