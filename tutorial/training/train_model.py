import config
import os

#       _______        _         __  __           _      _
#      |__   __|      (_)       |  \/  |         | |    | |
#         | |_ __ __ _ _ _ __   | \  / | ___   __| | ___| |
#         | | '__/ _` | | '_ \  | |\/| |/ _ \ / _` |/ _ \ |
#         | | | | (_| | | | | | | |  | | (_) | (_| |  __/ |
#         |_|_|  \__,_|_|_| |_| |_|  |_|\___/ \__,_|\___|_|
# - Now, we will put all the previous steps together to train the model



# ------------------------------------------------------------
# 1. Get the previously built items for training
# ------------------------------------------------------------

# 1.1 Get the built model
from tutorial.training.compile_model import model
model = model

# 1.2 Get the optimizer
from tutorial.training.optimizer import optimizer, jit_compile
optimizer = optimizer
jit_compile = jit_compile

# 1.3 Compile the built model
# - Here we connect the optimizer to the model
model.compile(optimizer=optimizer, jit_compile=jit_compile)

# 1.4 Get the datasets we will use for training
from tutorial.training.load_dataset import train_dataset, val_dataset
train_dataset = train_dataset
val_dataset = val_dataset



# ------------------------------------------------------------
# 2. Create model training checkpoints
# ------------------------------------------------------------
# - We would like to save the model after each training epoch if its validation accuracy improved
import tensorflow as tf

# 2.1 Define model save path
model_save_name = 'tutorial_model'
save_path = os.path.join(config.trained_models_dir, model_save_name)

# 2.2 Create the checkpoint object
# - We only save the model if validation accuracy improved
save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    save_path,
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True
)
model_checkpoints = [
    save_checkpoint
]




# ------------------------------------------------------------
# 3. Train the model
# ------------------------------------------------------------
# - Finally, we are ready to train our language model

# 3.1 Define the number of training epochs
epochs = 30


# 3.2 Start the training with the fit function
# - This function returns the training history of the model when finished
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=model_checkpoints
)



# ------------------------------------------------------------
# 4. Plot the training history
# ------------------------------------------------------------
# - We can now plot the training history with matplotlib
import matplotlib.pyplot as plt

# 4.1 Plot the training and validation loss
save_path = os.path.join(config.root_dir, 'tutorial', 'training', 'training_loss.png')
plt.figure(figsize=(10, 6))
plt.plot(history.history['perplexity'], label='Training Perplexity')
plt.plot(history.history['val_perplexity'], label='Validation Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.legend()
plt.show()
plt.savefig(save_path)



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try training different models changing the training parameters
# - Make sure to save the model with a different name each time
# - Play around with the training epochs, batch size, and learning rate to understand their effects

