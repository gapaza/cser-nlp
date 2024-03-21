import config
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from preprocessing.DatasetGenerator import DatasetGenerator
from model import get_requirement_decoder as get_model
import tensorflow_addons as tfa




def train():

    # 1. Build Model
    checkpoint_path = None
    model = get_model(checkpoint_path=checkpoint_path)

    # 2. Get Optimizer
    optimizer, jit_compile = get_optimizer()

    # 3. Compile Model
    model.compile(optimizer=optimizer, jit_compile=jit_compile)

    # 4. Get Datasets
    train_dataset, val_dataset = get_dataset()

    # 5. Get Checkpoints
    checkpoints = get_checkpoints()

    # 6. Train Model
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        callbacks=checkpoints
    )

    # 7. Plot History
    plot_history(history)



# """
#       _    _      _
#      | |  | |    | |
#      | |__| | ___| |_ __   ___ _ __ ___
#      |  __  |/ _ \ | '_ \ / _ \ '__/ __|
#      | |  | |  __/ | |_) |  __/ |  \__ \
#      |_|  |_|\___|_| .__/ \___|_|  |___/
#                    | |
#                    |_|
# """

def get_dataset():
    dg = DatasetGenerator(config.used_dataset)
    train_dataset, val_dataset = dg.load_datasets()
    return train_dataset, val_dataset

def get_optimizer():
    jit_compile = False
    learning_rate = 0.0001
    # learning_rate_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     0.0,  # initial learning rate
    #     1000,  # decay_steps
    #     alpha=1.0,
    #     warmup_target=learning_rate,
    #     warmup_steps=100
    # )
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate_scheduler)
    optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate)
    return optimizer, jit_compile

def get_checkpoints():
    checkpoints = []
    save_dir = config.model_path
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir, monitor='val_loss', save_weights_only=True, save_best_only=True)
    checkpoints.append(model_checkpoint)
    return checkpoints

def plot_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['perplexity'], label='Training Perplexity')
    plt.plot(history.history['val_perplexity'], label='Validation Perplexity')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    train()


