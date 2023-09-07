import keras
import tensorflow as tf
import numpy as np
from keras.initializers.initializers_v2 import Constant
from keras.models import load_model
from keras.utils import text_dataset_from_directory
from keras.layers import TextVectorization, Dropout, Dense, GlobalMaxPooling1D  # Embedding ,Bidirectional, LSTM
from keras.callbacks import ModelCheckpoint

from PositionalEmbedding import PositionalEmbedding
# from ModelBuilding import ModelBuilding
from TransformerEncoder import TransformerEncoder

# import re
# import string
# import math

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

batch_size = 32
seed = 1337

train_data_set = text_dataset_from_directory("aclImdb/train", batch_size=batch_size, seed=seed)

validation_data_set = text_dataset_from_directory("aclImdb/validation", batch_size=batch_size, seed=seed)

test_data_set = text_dataset_from_directory("aclImdb/test", batch_size=batch_size, seed=seed)

# for data_set, data_set_name in ((train_data_set, "train_data_set"),
#                                 (validation_data_set, "validation_data_set"),
#                                 (test_data_set, "test_data_set")):
#     for inputs, targets in data_set:
#         print("\n--- " + data_set_name + " ---")
#         print("inputs.shape:", inputs.shape)
#         print("inputs.dtype:", inputs.dtype)
#         print("targets.shape:", targets.shape)
#         print("targets.dtype:", targets.dtype)
#         print("inputs[0]:", inputs[0])
#         print("targets[0]:", targets[0])
#         break
#
#
# def custom_standardization_fn(string_tensor):
#     # Convert strings to lowercase
#     lowercase_string = tf.strings.lower(string_tensor)
#     # Replace punctuation characters with the empty string
#     return tf.strings.regex_replace(
#         lowercase_string, f"[{re.escape(string.punctuation)}]", "")
#
#
# def custom_split_fn(string_tensor):
#     # Split strings on whitespace
#     return tf.strings.split(string_tensor)
#
#
# def tfidf(term, document, dataset):
#     term_freq = document.count(term)
#     doc_freq = math.log(sum(doc.count(term) for doc in dataset) + 1)
#     return term_freq / doc_freq


# Configures the layer to return sequences of words encoded as integer indices.
# Standardization: convert to lowercase and remove punctuation
# Tokenization: split on whitespace
sequence_length = 600
max_tokens = 20000
text_vectorization = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",  # "multi_hot", "count", "tf_idf"
    output_sequence_length=sequence_length
    # ngrams=2,
    # standardize=custom_standardization_fn,
    # split=custom_split_fn
)

# Prepare a dataset that only yields raw text inputs (no labels)
text_only_train_data_set = train_data_set.map(lambda text_input, target: text_input)

# Index the dataset vocabulary via the adapt method
text_vectorization.adapt(text_only_train_data_set)

embedding_dim = 300

# Parsing the GloVe word-embeddings file
path_to_glove_file = "./glove.6B/glove.6B." + str(embedding_dim) + "d.txt"

embeddings_index = {}
with open(path_to_glove_file, encoding="utf8") as file:
    for line in file:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")

vocabulary = text_vectorization.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))

embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# Prepare processed versions of our training, validation, and test dataset
vectorized_ngram_train_data_set = train_data_set.map(lambda text, target: (text_vectorization(text), target),
                                                     num_parallel_calls=16)
vectorized_ngram_validation_data_set = validation_data_set.map(lambda text, target: (text_vectorization(text), target),
                                                               num_parallel_calls=16)
vectorized_ngram_test_data_set = test_data_set.map(lambda text, label: (text_vectorization(text), label),
                                                   num_parallel_calls=16)

# for data_set, data_set_name in ((binary_1gram_train_data_set, "binary_1gram_train_data_set"),
#                                 (binary_1gram_validation_data_set, "binary_1gram_validation_data_set"),
#                                 (binary_1gram_test_data_set, "binary_1gram_test_data_set")):
#     for inputs, targets in data_set:
#         print("\n--- " + data_set_name + " ---")
#         print("inputs.shape:", inputs.shape)
#         print("inputs.dtype:", inputs.dtype)
#         print("targets.shape:", targets.shape)
#         print("targets.dtype:", targets.dtype)
#         print("inputs[0]:", inputs[0])
#         print("targets[0]:", targets[0])
#         break

num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
# embedded = tf.one_hot(inputs, depth=max_tokens)
# embedded = Embedding(input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
# embedded = Embedding(max_tokens, embedding_dim)(inputs)
# embedded = Embedding(input_dim=max_tokens,
#                      output_dim=embedding_dim,
#                      embeddings_initializer=Constant(embedding_matrix),
#                      trainable=True,
#                      mask_zero=True)(inputs)
embedded = PositionalEmbedding(sequence_length=sequence_length,
                               input_dim=max_tokens,
                               output_dim=embedding_dim,
                               embeddings_initializer=Constant(embedding_matrix),
                               trainable=True,
                               mask_zero=True)(inputs)
x = TransformerEncoder(embedding_dim, dense_dim, num_heads)(embedded)
# x = Bidirectional(LSTM(32))(embedded)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
# model = ModelBuilding.get_model()

model.summary()

callbacks = [
    # ModelCheckpoint(str(text_vectorization._output_mode) + "_" + str(text_vectorization._ngrams_arg) + "gram.keras",
    #                 save_best_only=True)
    # ModelCheckpoint("one_hot_bidir_lstm.keras", save_best_only=True)
    # ModelCheckpoint("embeddings_bidir_gru_with_masking.keras", save_best_only=True)
    # ModelCheckpoint("glove_embeddings_sequence_model.keras", save_best_only=True)
    ModelCheckpoint("full_transformer_encoder.keras", save_best_only=True)
]

model.fit(vectorized_ngram_train_data_set.cache(),
          validation_data=vectorized_ngram_validation_data_set.cache(),
          epochs=10,
          callbacks=callbacks)

# model = load_model(str(text_vectorization._output_mode) + "_" + str(text_vectorization._ngrams_arg) + "gram.keras")
# model = load_model("one_hot_bidir_lstm.keras")
# model = load_model("embeddings_bidir_gru_with_masking.keras")
# model = load_model("glove_embeddings_sequence_model.keras")
model = load_model("full_transformer_encoder.keras", custom_objects={"TransformerEncoder": TransformerEncoder,
                                                                     "PositionalEmbedding": PositionalEmbedding,
                                                                     "Constant": Constant, })

print(f"\nTest acc: {model.evaluate(vectorized_ngram_test_data_set)[1]:.3f}")

# Exporting a model that processes raw strings
print("\nExporting a model that processes raw strings")
inputs = keras.Input(shape=(1,), dtype="string")
processed_inputs = text_vectorization(inputs)
outputs = model(processed_inputs)
inference_model = keras.Model(inputs, outputs)

raw_text_data = tf.convert_to_tensor([
    ["That was an excellent movie, I loved it."],
    ["What a lame movie. I hated it!"],
])

predictions = inference_model(raw_text_data)

for prediction in predictions:
    print(f"{float(prediction * 100):.2f} percent positive")

# dataset = [
#     "I write, erase, rewrite",
#     "Erase again, and then",
#     "A poppy blooms."
# ]
# test_sentence = "I write, rewrite, and still rewrite again"
#
# # Index the vocabulary
# text_vectorization.adapt(dataset)
# vocabulary = text_vectorization.get_vocabulary()
# print(vocabulary)
#
# inverse_vocabulary = dict(enumerate(vocabulary))
# print(inverse_vocabulary)
#
# encoded_sentence = text_vectorization(test_sentence)
# print(encoded_sentence)
#
# decoded_sentence = " ".join(inverse_vocabulary[int(index)] for index in encoded_sentence)
# print(decoded_sentence)


# from Vectorizer import Vectorizer


# vectorizer = Vectorizer()
# vectorizer.make_vocabulary(dataset)
#
# encoded_sentence = vectorizer.encode(test_sentence)
# decoded_sentence = vectorizer.decode(encoded_sentence)
#
# print(dataset)
# print(test_sentence)
# print(encoded_sentence)
# print(decoded_sentence)
