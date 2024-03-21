import keras
from keras import layers
import tensorflow as tf
import config
import keras_nlp
import math
from keras_nlp.layers import TransformerDecoder
from keras_nlp.layers import TokenAndPositionEmbedding


from model.vocabulary import max_seq_len, vocab_size


# ------------------------------------
# Actor
# ------------------------------------

@keras.saving.register_keras_serializable(package="ConditionalRequirementDecoder", name="ConditionalRequirementDecoder")
class ConditionalRequirementDecoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

        self.vocab_size = vocab_size
        self.vocab_output_size = vocab_size
        self.gen_design_seq_length = max_seq_len

        self.embed_dim = 256
        self.num_heads = 2
        self.dense_dim = 256
        self.dropout = 0.0

        # Token + Position embedding
        self.requirement_embedding_layer = TokenAndPositionEmbedding(
            self.vocab_size,
            self.gen_design_seq_length,
            self.embed_dim,
            mask_zero=True
        )

        # Decoder Stack
        self.normalize_first = False
        self.decoder_1 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_1', dropout=self.dropout)
        # self.decoder_2 = TransformerDecoder(self.dense_dim, self.num_heads, normalize_first=self.normalize_first, name='decoder_2', dropout=self.dropout)

        # Requirement Prediction Head
        self.requirement_prediction_head = layers.Dense(
            self.vocab_output_size,
            name="design_prediction_head"
        )
        self.activation = layers.Activation('softmax', dtype='float32')
        self.log_activation = layers.Activation('log_softmax', dtype='float32')

    def call(self, inputs, training=True, mask=None):
        requirements = inputs

        # 1. Embed requirements
        requirements_embedded = self.requirement_embedding_layer(requirements, training=training)

        # 2. Decoder Stack
        decoded_requirements = requirements_embedded
        decoded_requirements = self.decoder_1(decoded_requirements, use_causal_mask=True, training=training)
        # decoded_requirements = self.decoder_2(decoded_requirements, use_causal_mask=True, training=training)

        # 3. Requirement Prediction Head
        requirements_prediction_logits = self.requirement_prediction_head(decoded_requirements)
        requirements_prediction = self.activation(requirements_prediction_logits)

        return requirements_prediction  # For training

    def get_config(self):
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    # ------------------------------------
    # Training Loop
    # ------------------------------------

    # Token Prediction
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    perplexity_tracker = keras_nlp.metrics.Perplexity(mask_token_id=0)
    loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, inputs):
        requirements_input, target_requirements = inputs

        with tf.GradientTape() as tape:
            predictions = self(requirements_input, training=True)
            uloss = self.loss_fn(target_requirements, predictions)
            if config.mixed_precision is True:
                loss = self.optimizer.get_scaled_loss(uloss)
            else:
                loss = uloss

        gradients = tape.gradient(loss, self.trainable_variables)
        if config.mixed_precision is True:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(uloss)
        self.perplexity_tracker.update_state(target_requirements, predictions)
        return {
            "loss": self.loss_tracker.result(),
            "perplexity": self.perplexity_tracker.result()
        }

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

    @property
    def metrics(self):
        return [self.loss_tracker, self.perplexity_tracker]
















