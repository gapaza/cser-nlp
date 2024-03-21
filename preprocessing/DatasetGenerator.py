import config
import tensorflow as tf
import random
from tqdm import tqdm
from multiprocessing import Pool
import os
import model.vocabulary as vocabulary


@tf.function
def encode_batch(requirements_inputs, requirements_targets):
    requirements_inputs_encoded = vocabulary.encode_tf(requirements_inputs)
    requirements_targets_encoded = vocabulary.encode_tf(requirements_targets)
    return requirements_inputs_encoded, requirements_targets_encoded



class DatasetGenerator:

    def __init__(self, dataset_dir):
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.train_dataset_dir = os.path.join(self.dataset_dir, 'train_dataset')
        self.val_dataset_dir = os.path.join(self.dataset_dir, 'val_dataset')

    # """
    #       _____
    #      |  __ \
    #      | |__) | __ ___ _ __   __ _ _ __ ___
    #      |  ___/ '__/ _ \ '_ \ / _` | '__/ _ \
    #      | |   | | |  __/ |_) | (_| | | |  __/
    #      |_|   |_|  \___| .__/ \__,_|_|  \___|
    #                     | |
    #                     |_|
    # """

    def prepare_datasets(self):

        # 1. Load requirements lines
        req_data = config.req_data
        req_data = list(set(req_data))

        # 2. Preprocess datapoints
        processed_datapoints = []
        with tqdm(total=len(req_data)) as progress_bar:
            for requirement in req_data:
                processed_datapoints.append(DatasetGenerator.preprocess_datapoint(requirement))
                progress_bar.update(1)
        random.shuffle(processed_datapoints)

        # 3. Split into train and validation
        split_idx = int(0.95 * len(processed_datapoints))
        train_datapoints = processed_datapoints[:split_idx]
        val_datapoints = processed_datapoints[split_idx:]

        # 4. Create dataset
        train_dataset = self.create_dataset(train_datapoints)
        val_dataset = self.create_dataset(val_datapoints)

        # 5. Save dataset
        train_dataset.save(self.train_dataset_dir)
        val_dataset.save(self.val_dataset_dir)

    @staticmethod
    def preprocess_datapoint(requirement):
        requirement_input = vocabulary.start_token_label + ' ' + requirement
        requirement_target = requirement + ' ' + vocabulary.end_token_label
        return [requirement_input, requirement_target]


    def create_dataset(self, datapoints):
        requirements_inputs = [datapoint[0] for datapoint in datapoints]
        requirements_targets = [datapoint[1] for datapoint in datapoints]
        dataset = tf.data.Dataset.from_tensor_slices(
            (requirements_inputs, requirements_targets)
        )
        dataset = dataset.batch(config.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    # """
    #       _                     _
    #      | |                   | |
    #      | |     ___   __ _  __| |
    #      | |    / _ \ / _` |/ _` |
    #      | |___| (_) | (_| | (_| |
    #      |______\___/ \__,_|\__,_|
    #
    # """

    def load_datasets(self, batch_size=None):
        train_path = self.train_dataset_dir
        val_path = self.val_dataset_dir
        if batch_size is not None:
            train_path = train_path + '_' + str(batch_size)
            val_path = val_path + '_' + str(batch_size)
        train_dataset = tf.data.Dataset.load(train_path)
        val_dataset = tf.data.Dataset.load(val_path)
        return train_dataset, val_dataset

    def rebatch_dataset(self, batch=config.batch_size):
        train_dataset, val_dataset = self.load_datasets()
        train_dataset = train_dataset.rebatch(batch)
        val_dataset = val_dataset.rebatch(batch)
        train_dataset.save(self.train_dataset_dir + '_' + str(batch))
        val_dataset.save(self.val_dataset_dir + '_' + str(batch))




if __name__ == '__main__':
    dg = DatasetGenerator(config.used_dataset)
    dg.prepare_datasets()

