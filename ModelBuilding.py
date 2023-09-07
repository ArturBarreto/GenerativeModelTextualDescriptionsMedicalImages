import keras
from keras import layers


class ModelBuilding:

    def get_model(self, max_tokens=20000, hidden_dim=16):
        inputs = keras.Input(shape=(max_tokens,))
        x = layers.Dense(hidden_dim, activation="relu")(inputs)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        return model
