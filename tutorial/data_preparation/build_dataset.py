import config

#       ____        _ _     _   _____        _                 _
#      |  _ \      (_) |   | | |  __ \      | |               | |
#      | |_) |_   _ _| | __| | | |  | | __ _| |_ __ _ ___  ___| |_
#      |  _ <| | | | | |/ _` | | |  | |/ _` | __/ _` / __|/ _ \ __|
#      | |_) | |_| | | | (_| | | |__| | (_| | || (_| \__ \  __/ |_
#      |____/ \__,_|_|_|\__,_| |_____/ \__,_|\__\__,_|___/\___|\__|
# - Now it is time to put it all together and build or dataset to train our model on
print_output = True

# -----------------------------------------------
# 1. Preprocess the input/target pairs
# -----------------------------------------------
# For each requirement, we will create an input and target pair
# - The input datapoint will be fed into the model during training
# - The target datapoint will be used to calculate the loss
# For training, we need to inform the model about the start and end of each requirement
# - For inputs, we add a start token at the beginning to inform the model that it is the start of the requirement
# - For targets, we add an end token at the end to inform the model that it is the end of the requirement
# The purpose of this will be clear when we conduct inference later on

# 1.1. Import the list of requirements we created previously, start token, and end token
from tutorial.data_preparation.build_vocabulary import req_data, start_token_label, end_token_label

# 1.2. Iterate over the list, and create an input and target datapoint for each requirement
preprocessed_pairs = []
for requirement in req_data:
    requirement_input = start_token_label + ' ' + requirement
    requirement_target = requirement + ' ' + end_token_label
    preprocessed_pairs.append([requirement_input, requirement_target])

# 1.3. We can shuffle the pairs to ensure that subsystem requirements are not grouped together
import random
random.shuffle(preprocessed_pairs)


# -----------------------------------------------
# 2. Split the dataset into training and validation
# -----------------------------------------------

# 2.1 Determine what percent of the dataset to use for training
train_percent = 0.95

# 2.2. Split the dataset into training and validation
split_idx = int(train_percent * len(preprocessed_pairs))
train_pairs = preprocessed_pairs[:split_idx]
val_pairs = preprocessed_pairs[split_idx:]

# 2.3. Print the number of training and validation pairs
if print_output:
    print('\n\nSection 1: Dataset Split')
    print('---- Number of training pairs: ', len(train_pairs))
    print('-- Number of validation pairs: ', len(val_pairs))


# -----------------------------------------------
# 3. Create TensorFlow datasets
# -----------------------------------------------
# For this step, we will use the tensorflow Dataset API to create datasets for training and validation
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
import tensorflow as tf
batch_size = 128

# 3.1. Create a tensorflow dataset for the training pairs, and batch them
train_requirements_inputs = [pair[0] for pair in train_pairs]
train_requirements_targets = [pair[1] for pair in train_pairs]
train_dataset_text = tf.data.Dataset.from_tensor_slices(
    (train_requirements_inputs, train_requirements_targets)
)
train_dataset_text = train_dataset_text.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)

# 3.2. Create a tensorflow dataset for the validation pairs, and batch them
val_requirements_inputs = [pair[0] for pair in val_pairs]
val_requirements_targets = [pair[1] for pair in val_pairs]
val_dataset_text = tf.data.Dataset.from_tensor_slices(
    (val_requirements_inputs, val_requirements_targets)
)
val_dataset_text = val_dataset_text.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)


# -----------------------------------------------
# 4. Tokenize the datasets
# -----------------------------------------------
# The datasets currently contain text data, we need to convert them into token ids
# To accomplish this, we can use the tensorflow dataset map function
# - This function allows us to apply a custom transformation to each element of the dataset

# 4.1. Define the custom transformation function
from tutorial.data_preparation.tokenize import encode_tf
@tf.function
def encode_batch(requirements_inputs, requirements_targets):
    requirements_inputs_encoded = encode_tf(requirements_inputs)
    requirements_targets_encoded = encode_tf(requirements_targets)
    return requirements_inputs_encoded, requirements_targets_encoded

# 4.2. Apply the transformation to the training and validation datasets
train_dataset = train_dataset_text.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset_text.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)



# -----------------------------------------------
# 5. Save the datasets
# -----------------------------------------------
import os

# 5.1 Define the dataset name and paths
dataset_name = 'tutorial_dataset'
dataset_path = os.path.join(config.datasets_dir, dataset_name)
train_dataset_path = os.path.join(dataset_path, 'train_dataset')
val_dataset_path = os.path.join(dataset_path, 'val_dataset')

# 5.2 Save the datasets
train_dataset.save(train_dataset_path)
val_dataset.save(val_dataset_path)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# Datasets API: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
# - Try creating different dataset by uncommenting or commenting out requirements files used in the build_vocabulary file
# - NOTE: make sure to re-name the datasets with the dataset_name variable
# - Try sampling points from the dataset using the tensorflow .take() function (view the dataset API)
# - Try shuffling the dataset using the shuffle function (view the dataset API)
# - Try determining how many elements are in the datasets you created using the cardinality function (view the dataset API)
# YOUR EXPERIMENTAL CODE HERE











