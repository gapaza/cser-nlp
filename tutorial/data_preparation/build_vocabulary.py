import sys
import os
from pathlib import Path
curr_path = Path(os.path.dirname(os.path.abspath(__file__)))
root_path = curr_path.parents[1]  # parents[0] is one directory up, parents[1] is two directories up
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
import config

#       ____        _ _     _  __      __             _           _
#      |  _ \      (_) |   | | \ \    / /            | |         | |
#      | |_) |_   _ _| | __| |  \ \  / /__   ___ __ _| |__  _   _| | __ _ _ __ _   _
#      |  _ <| | | | | |/ _` |   \ \/ / _ \ / __/ _` | '_ \| | | | |/ _` | '__| | | |
#      | |_) | |_| | | | (_| |    \  / (_) | (_| (_| | |_) | |_| | | (_| | |  | |_| |
#      |____/ \__,_|_|_|\__,_|     \/ \___/ \___\__,_|_.__/ \__,_|_|\__,_|_|   \__, |
#                                                                               __/ |
#                                                                              |___/
print_output = False


# -----------------------------------------------------
# 1. Load all requirement statements into a list
# -----------------------------------------------------

# 1.1. Requirement files are stored in the 'datafiles' directory
datafiles_dir = config.datafiles_dir

# 1.2. These are the files we will parse for requirements
req_files = [
    'adcs_requirements.txt',
    # 'gs_requirements.txt',
    # 'comm_requirements.txt',
    # 'power_requirements.txt',
    # 'thermal_requirements.txt',
    # 'obdh_requirements.txt',
]

# 1.3. Iterate over the files, read the requirements and store them in a list
req_data = []
for req_file in req_files:
    file_path = os.path.join(datafiles_dir, req_file)
    with open(file_path, 'r') as f:
        requirements = f.readlines()
    requirements = [req.strip() for req in requirements]
    requirements = list(set(requirements))
    if '' in requirements:
        requirements.remove('')
    req_data.extend(requirements)

# 1.4. Count the number of words in the requirements
total_word_count = 0
for req in req_data:
    total_word_count += len(req.split())

# Relevant Output Information
if print_output:
    print('\n\nSection 1: Requirement Data')
    print('------------ Sample requirement: ', req_data[0])
    print('-- Total number of requirements: ', len(req_data))
    print('--------- Total number of words: ', total_word_count)
    print('- Average words per requirement: ', total_word_count / len(req_data))


# -----------------------------------------------------
# 2. Create a vocabulary from the requirements
# -----------------------------------------------------
import tensorflow as tf
from keras.layers import TextVectorization

# 2.1. Define the maximum sequence length and maximum number of tokens
max_seq_len = 50
max_tokens = 20000

# 2.2. Create a TextVectorization layer to tokenize the requirements
# - The purpose of this layer is to take in raw text data and convert it into a sequence of integers (tokens)
# https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/
tokenizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_seq_len,
)

# 2.3. Adapt the tokenizer to the requirements data
# - The adapt function analyzes the data, and creates a vocabulary index based on word frequency
tokenizer.adapt(req_data)

# 2.4. Get the vocabulary and add special tokens that we will use when creating the dataset
# Special Tokens
# - [start]: Indicates the start of a sequence
# - [end]: Indicates the end of a sequence
vocabulary = tokenizer.get_vocabulary()
vocabulary.extend(["[start]", "[end]"])
tokenizer.set_vocabulary(vocabulary)
vocabulary = tokenizer.get_vocabulary()
vocab_size = len(vocabulary)

# 2.5. Create a mapping between token ids and tokens for easy lookup
id2token = dict(enumerate(vocabulary))
token2id = {y: x for x, y in id2token.items()}

# 2.6. Record the token ids for the special tokens, this will simplify the dataset creation process
start_token_label = '[start]'
end_token_label = '[end]'
start_token_id = tokenizer(['[start]']).numpy()[0][0]
end_token_id = tokenizer(['[end]']).numpy()[0][0]

# Relevant Output Information
if print_output:
    print('\n\nSection 2: Vocabulary Creation')
    print('--------------- Vocabulary Size: ', vocab_size)
    print('-- Vocabulary (first 10 tokens): ', vocabulary[:10])



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try uncommenting some of the requirements files used to see how the size of the vocabulary changes
# - Try seeing which words around found in the vocabulary, are there any missing that surprise you?
# YOUR EXPERIMENTAL CODE HERE













