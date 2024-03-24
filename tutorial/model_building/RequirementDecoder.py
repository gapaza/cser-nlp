import config

#   _____       _   _   _               _ _           _ _   _______               _   _
#  |  __ \     | | | | (_)             (_) |         | | | |__   __|             | | | |
#  | |__) |   _| |_| |_ _ _ __   __ _   _| |_    __ _| | |    | | ___   __ _  ___| |_| |__   ___ _ __
#  |  ___/ | | | __| __| | '_ \ / _` | | | __|  / _` | | |    | |/ _ \ / _` |/ _ \ __| '_ \ / _ \ '__|
#  | |   | |_| | |_| |_| | | | | (_| | | | |_  | (_| | | |    | | (_) | (_| |  __/ |_| | | |  __/ |
#  |_|    \__,_|\__|\__|_|_| |_|\__, | |_|\__|  \__,_|_|_|    |_|\___/ \__, |\___|\__|_| |_|\___|_|
#                                __/ |                                  __/ |
#                               |___/                                  |___/
# - We will now combine all the layers we created to create a Keras model class
# - To do so, we will subclass the tf.keras.Model class

from tutorial.model_building.embedding_layer import embedding_layer
from tutorial.model_building.decoder_layer import decoder_layer
from tutorial.model_building.output_layer import output_layer, activation_layer


# ------------------------------------
# Model
# ------------------------------------
import keras
import tensorflow as tf
import keras_nlp


@keras.saving.register_keras_serializable(package="RequirementDecoder", name="RequirementDecoder")
class RequirementDecoder(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # This is required for the model to support the masking of padding tokens
        self.supports_masking = True

        # Embedding layer
        self.embedding_layer = embedding_layer

        # Decoder layer
        self.decoder_layer = decoder_layer

        # Output layer
        self.output_layer = output_layer
        self.activation_layer = activation_layer



    # ------------------------------------------
    # Call method
    # ------------------------------------------
    # - This is the method we will call to pass inputs through the model
    # - It is where we define the forward pass of the model
    def call(self, inputs, training=True, mask=None):
        x = inputs

        # Pass the inputs through the embedding layer
        x = self.embedding_layer(x, training=training)

        # Pass the embedded inputs through the decoder layer
        x = self.decoder_layer(x, use_causal_mask=True, training=training)

        # Pass the decoder output through the output layer
        x = self.output_layer(x)

        # Apply the softmax activation to the output
        x = self.activation_layer(x)

        return x


    # ------------------------------------------
    # Config methods
    # ------------------------------------------
    # - These are less important for now, but are required for saving and loading the model
    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # ------------------------------------
    # Loss Functions
    # ------------------------------------

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=0)
    perplexity_tracker = keras_nlp.metrics.Perplexity(mask_token_id=0)
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    # ------------------------------------
    # Training Step
    # ------------------------------------
    # - Here, we define the training step for the model
    # - This is where we calculate the loss and update the model weights
    # - This is where BACKPROPAGATION happens

    def train_step(self, inputs):
        requirements_input, target_requirements = inputs

        with tf.GradientTape() as tape:
            predictions = self(requirements_input, training=True)
            loss = self.loss_fn(target_requirements, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(target_requirements, predictions)
        return {
            "loss": self.loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result()
        }


    # ------------------------------------
    # Testing Step
    # ------------------------------------
    # - Here, we define the testing step for the model
    # - This function is called when we are testing the model on a validation set
    # - No backpropagation happens here, and thus it is much faster than the training step

    def test_step(self, inputs):
        requirements_input, target_requirements = inputs
        predictions = self(requirements_input, training=False)
        loss = self.loss_fn(target_requirements, predictions)
        self.loss_tracker.update_state(loss)
        self.perplexity_tracker.update_state(target_requirements, predictions)
        return {
            "loss": self.loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result()
        }

    # ------------------------------------
    # Metrics
    # ------------------------------------
    # - These are the metrics that will be displayed during training and testing

    @property
    def metrics(self):
        return [self.loss_tracker, self.perplexity_tracker]



#                    _   _       _ _
#          /\       | | (_)     (_) |
#         /  \   ___| |_ ___   ___| |_ _   _
#        / /\ \ / __| __| \ \ / / | __| | | |
#       / ____ \ (__| |_| |\ V /| | |_| |_| |
#      /_/    \_\___|\__|_| \_/ |_|\__|\__, |
#                                       __/ |
#                                      |___/
# - Try instantiating the model class defined above
# - Try executing a forward pass of the model using a sample encoded requirement
# - NOTE: we cannot call the train or test step without compiling the model first
# --- Modify the one provided below to your liking
from tutorial.data_preparation.build_vocabulary import start_token_label, end_token_label
from tutorial.data_preparation.tokenize import encode_tf
input_text = start_token_label + ' ' + 'The ADCS shall'
label_text = 'The ADCS shall' + ' ' + end_token_label



