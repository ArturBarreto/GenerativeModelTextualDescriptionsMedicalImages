from keras import layers
import tensorflow as tf


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, embeddings_initializer, trainable, mask_zero, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim, embeddings_initializer=embeddings_initializer,
            trainable=trainable, mask_zero=mask_zero)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.trainable = trainable
        self.mask_zero = mask_zero

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "embeddings_initializer": self.embeddings_initializer,
            "trainable": self.trainable,
            "mask_zero": self.mask_zero,
        })
        return config
