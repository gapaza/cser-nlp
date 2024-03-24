import config
import os

#       _                     _   _____        _                 _
#      | |                   | | |  __ \      | |               | |
#      | |     ___   __ _  __| | | |  | | __ _| |_ __ _ ___  ___| |_
#      | |    / _ \ / _` |/ _` | | |  | |/ _` | __/ _` / __|/ _ \ __|
#      | |___| (_) | (_| | (_| | | |__| | (_| | || (_| \__ \  __/ |_
#      |______\___/ \__,_|\__,_| |_____/ \__,_|\__\__,_|___/\___|\__|
# - For this file, we will focus on loading the dataset we previously saved



# -----------------------------------------------------
# 1. Load the dataset
# -----------------------------------------------------
import tensorflow as tf

# 1.1. Get dataset paths
# - We defined these paths when we built the dataset, but we will redefine them in case we want a different dataset
dataset_name = 'tutorial_dataset_2'
dataset_path = os.path.join(config.datasets_dir, dataset_name)
train_dataset_path = os.path.join(dataset_path, 'train_dataset')
val_dataset_path = os.path.join(dataset_path, 'val_dataset')

# 1.2. Load the datasets
# - We load the datasets using the tensorflow Dataset API
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset
train_dataset = tf.data.Dataset.load(train_dataset_path)
val_dataset = tf.data.Dataset.load(val_dataset_path)



# -----------------------------------------------------
# 2. Rebatch the datasets (optional)
# -----------------------------------------------------
# - If you would like to use a different batch size, you can re-batch the datasets
# - This is useful if your computer has limited memory, or if you want to experiment with different batch sizes
# - NOTE: If the dataset is rebatched, tensorflow will not know the size of the dataset during training
new_batch_size = 64
# train_dataset = train_dataset.rebatch(new_batch_size)
# val_dataset = val_dataset.rebatch(new_batch_size)


# -----------------------------------------------------
# 3. Define shuffle operations and prefetching
# -----------------------------------------------------

# 3.1. Shuffle the datasets
# - This function is re-shuffling the dataset for each epoch
# - We only shuffle the training dataset
# - The buffer size is the number of dataset elements to shuffle at a time
buffer_size = 128
train_dataset = train_dataset.shuffle(buffer_size=buffer_size)

# 3.2. Prefetch the datasets
# - This function is prefetching elements from the dataset to speed up training
# - Tensorflow will prefetch the next batch of elements while the current batch is being processed!
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try loading any different datasets you created by changing the dataset_name variable
# - Try changing the batch size and seeing how the dataset elements differ
# - Try shuffling the dataset with different buffer sizes
# --- Try taking elements from the dataset after these changes and printing them to see how they differ
# YOUR EXPERIMENTAL CODE HERE



