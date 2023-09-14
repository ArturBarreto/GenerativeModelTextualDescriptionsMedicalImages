###########################
#          Setup          #
###########################
import datetime
import sys
from os import makedirs, environ, path
import re
from os.path import dirname

from sys import maxsize
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, VGG16, VGG19, DenseNet121, DenseNet201, InceptionV3
from tensorflow.keras.applications import InceptionResNetV2, Xception, ResNet50V2, ResNet101V2, ResNet152V2, NASNetMobile
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import TextVectorization
from time import time

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)

np.set_printoptions(threshold=sys.maxsize)

###########################
#         Config          #
###########################

# Seed to reproducibility
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Path to the images
IMAGES_PATH = "ImagesDataSet"

# Desired image dimensions
IMAGE_SIZE = (500, 500)

# Vocabulary size
VOCAB_SIZE = 50

# Fixed length allowed for any sequence
SEQ_LENGTH = 5

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Num of attentions heads
ENCODER_NUM_HEADS = 2
DECODER_NUM_HEADS = 2

# Base Model Trainable
BASE_MODEL_TRAINABLE = True

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 20
AUTOTUNE = tf.data.AUTOTUNE

# Num Captions per Image
NUM_CAPTIONS_PER_IMAGE = 1

###########################
#  Preparing the dataset  #
###########################


def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    with open(filename, encoding="utf-8") as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()
        sum_skipped = 0

        for line in caption_data:
            line = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = line.split("\t")

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            img_name = img_name.split("#")[0]
            img_name = path.join(IMAGES_PATH, img_name.strip())

            # We will remove caption that are either too short to too long
            tokens = caption.strip().split()

            if len(tokens) < 1 or len(tokens) > SEQ_LENGTH:
                images_to_skip.add(img_name)
                print("Pulei: " + img_name)
                sum_skipped = sum_skipped + 1
                continue

            if img_name.endswith("jpg") and img_name not in images_to_skip:
                # We will add a start and an end token to each caption
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        for img_name in images_to_skip:
            if img_name in caption_mapping:
                del caption_mapping[img_name]

        print("Soma de pulos: " + str(sum_skipped / NUM_CAPTIONS_PER_IMAGE))
        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data


####################### MAIN ############################
# Load the dataset
captions_mapping, text_data = load_captions_data("TextDataSet.txt")

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))
#########################################################


###########################
#Vectorizing the text data#
###########################

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)

    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")

    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


####################### MAIN ############################
vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

vectorization.adapt(text_data)
#########################################################


###########################
#  Pipeline for training  #
###########################


def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset


# Pass the list of images and the list of corresponding captions
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))


###########################
#   Building the model    #
###########################


def get_cnn_model(base_model):
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    cnn_model.trainable = base_model.trainable
    return cnn_model


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.0
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense_1 = layers.Dense(embed_dim, activation="relu")  # era embed_dim

    def call(self, inputs, training, mask=None):
        inputs = self.layernorm_1(inputs)
        inputs = self.dense_1(inputs)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=None,
            training=training,
        )
        out_1 = self.layernorm_2(inputs + attention_output_1)
        return out_1


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_scale = tf.math.sqrt(tf.cast(embed_dim, tf.float32))

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_tokens = embedded_tokens * self.embed_scale
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.ffn_layer_1 = layers.Dense(ff_dim, activation="relu")
        self.ffn_layer_2 = layers.Dense(embed_dim)

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(
            embed_dim=EMBED_DIM, sequence_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE
        )
        self.out = layers.Dense(VOCAB_SIZE, activation="softmax")

        self.dropout_1 = layers.Dropout(0.3)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=combined_mask,
            training=training,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
            training=training,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        ffn_out = self.layernorm_3(ffn_out + out_2, training=training)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model, encoder, decoder, num_captions_per_image, image_aug=None,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image
        self.image_aug = image_aug

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, img_embed, batch_seq, training=True):
        encoder_out = self.encoder(img_embed, training=training)
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out, training=training, mask=mask)
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        if self.image_aug:
            batch_img = self.image_aug(batch_img)

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass the caption to the decoder
        # along with the encoder output and compute the loss as well as accuracy
        # for the caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                loss, acc = self._compute_caption_loss_and_acc(
                    img_embed, batch_seq[:, i, :], training=True
                )

                # 3. Update loss and accuracy
                batch_loss += loss
                batch_acc += acc

            # 4. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 5. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 6. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        # 7. Update the trackers
        batch_acc /= float(self.num_captions_per_image)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 8. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass the caption to the decoder
        # along with the encoder output and compute the loss as well as accuracy
        # for the caption.
        for i in range(self.num_captions_per_image):
            loss, acc = self._compute_caption_loss_and_acc(
                img_embed, batch_seq[:, i, :], training=False
            )

            # 3. Update batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        batch_acc /= float(self.num_captions_per_image)

        # 4. Update the trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        # 5. Return the loss and accuracy values
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]


# Data augmentation for image data
def image_augmentation():
    image_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.3),
        ]
    )
    return image_augmentation


# Learning Rate Scheduler for the optimizer
class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, post_warmup_learning_rate, warmup_steps):
        super().__init__()
        self.post_warmup_learning_rate = post_warmup_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        global_step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_progress = global_step / warmup_steps
        warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
        return tf.cond(
            global_step < warmup_steps,
            lambda: warmup_learning_rate,
            lambda: self.post_warmup_learning_rate,
        )


vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 2
valid_images = list(valid_data.keys())

print(index_lookup)

confusion_matrix_type_exam = np.zeros((len(vocab), len(vocab)))
confusion_matrix_body_part = np.zeros((len(vocab), len(vocab)))
confusion_matrix_problem_type = np.zeros((len(vocab), len(vocab)))

val_list = list(index_lookup.values())

print(val_list)


def generate_caption(image_index):
    # Select a random image from the validation dataset
    sample_img = valid_images[image_index]
    sample_img_name = sample_img

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    # plt.imshow(img)
    # plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img, training=False, mask=mask)
        sampled_token_index = np.argmax(predictions[0, i, :])
        if sampled_token_index >= len(index_lookup):
            sampled_token_index = 1
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    # print("Imagem: " + sample_img_name)
    # print("Predicted: ", decoded_caption)
    label = str(captions_mapping.get(sample_img_name)).replace("<start> ", "")
    label = label.replace(" <end>", "")
    label = label.replace("['", "")
    label = label.replace("']", "").strip()
    # print("Label: ", label, "\n")
    return ("Imagem: " + sample_img_name, "Predicted: " + decoded_caption, "Label: " + label + "\n")


feature_extractor_models_names = ["VGG19"]  # "MobileNetV2", "DenseNet201", "ResNet152V2", "NASNetLarge", "X", "Xception", "InceptionV3", "InceptionResNetV2"]
                                            # Excluidos: "VGG16", "ResNet50V2", "ResNet101V2", "NASNetMobile", "DenseNet121"

optimizers_names = ["Adam"]  # , "AdamW", "Adadelta", "Adafactor"]
                   #"SGD", "RMSprop", "Adagrad", "Adamax", "Nadam", "Ftrl"]
weights_inicializer = ['imagenet']  # [None, 'imagenet']
trainables = [False]  # , True]
image_augmentations = [image_augmentation()]  # [None, image_augmentation()]

num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15

count_variation = 0
total_variations = 1  # len(feature_extractor_models_names) * len(optimizers_names) * len(weights_inicializer) * len(trainables) * len(image_augmentations) - len(feature_extractor_models_names) * len(optimizers_names) * len(image_augmentations)

agora = str(datetime.datetime.now()).replace(":", "-").replace(" ", "-")
output_path = "c:/Users/artur/OneDrive/Mestrado/ImageCaptioning/" + agora + "/"
makedirs(dirname(output_path), exist_ok=True)

output_all_path = output_path + "output_all.txt"
output_loss_path = output_path + "output_loss.txt"
output_acc_path = output_path + "output_acc.txt"
output_val_loss_path = output_path + "output_val_loss.txt"
output_val_acc_path = output_path + "output_val_acc.txt"
output_modelo_final_path = output_path + "output_modelo_final.txt"
output_confusion_matrix_type_exam_path = output_path + "output_confusion_matrix_type_exam.txt"
output_confusion_matrix_body_part_path = output_path + "output_confusion_matrix_body_part.txt"
output_confusion_matrix_problem_type_path = output_path + "output_confusion_matrix_problem_type.txt"

for weight in weights_inicializer:
    for trainable in trainables:

        if weight is None and trainable is False:
            continue

        # mobileNetV2 = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # mobileNetV2.trainable = trainable

        # denseNet201 = DenseNet201(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # denseNet201.trainable = trainable

        # resNet152V2 = ResNet152V2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # resNet152V2.trainable = trainable

        # nasnetLarge = NASNetLarge(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # nasnetLarge.trainable = trainable

        vgg19 = VGG19(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        vgg19.trainable = trainable

        # xception = Xception(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # xception.trainable = trainable

        # inceptionV3 = InceptionV3(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # inceptionV3.trainable = trainable

        # inceptionResNetV2 = InceptionResNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # inceptionResNetV2.trainable = trainable

        # vgg16 = VGG16(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # vgg16.trainable = trainable

        # resNet50V2 = ResNet50V2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # resNet50V2.trainable = trainable

        # resNet101V2 = ResNet101V2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # resNet101V2.trainable = trainable

        # nasnetMobile = NASNetMobile(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # nasnetMobile.trainable = trainable

        # denseNet121 = DenseNet121(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights=weight)
        # denseNet121.trainable = trainable

        feature_extractor_models = [vgg19]  # mobileNetV2, denseNet201, resNet152V2, nasnetLarge, X, xception, inceptionV3, inceptionResNetV2
                                            # Excluidos: vgg16, resNet50V2, resNet101V2, nasnetMobile, denseNet121

        for feature_extractor_model in feature_extractor_models:
            for image_augmentation in image_augmentations:
                ####################### MAIN ############################
                cnn_model = get_cnn_model(feature_extractor_model)
                encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, num_heads=ENCODER_NUM_HEADS)
                decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=DECODER_NUM_HEADS)
                caption_model = ImageCaptioningModel(
                    cnn_model=cnn_model,
                    encoder=encoder,
                    decoder=decoder,
                    num_captions_per_image=NUM_CAPTIONS_PER_IMAGE,
                    image_aug=image_augmentation
                )
                #########################################################


                ###########################
                #     Model training      #
                ###########################

                # Define the loss function
                # EarlyStopping criteria
                # Create a learning rate schedule

                adam = keras.optimizers.Adam(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # adamW = keras.optimizers.AdamW(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # adadelta = keras.optimizers.Adadelta(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # adafactor = keras.optimizers.Adafactor(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))

                # sgd = keras.optimizers.SGD(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # rmsprop = keras.optimizers.RMSprop(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # adagrad = keras.optimizers.Adagrad(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # adamax = keras.optimizers.Adamax(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # nadam = keras.optimizers.Nadam(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))
                # ftrl = keras.optimizers.Ftrl(LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps))

                optimizers = [adam]  # , adamW, adadelta, adafactor]  # Excluidos:  sgd, rmsprop, adagrad, adamax, nadam, ftrl

                for optimizer in optimizers:
                    # Compile the model
                    caption_model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none"))

                    # Fit the model

                    count_variation = count_variation + 1

                    with open(file=output_all_path, mode="a", encoding="utf-8") as output_all, \
                         open(file=output_loss_path, mode="a", encoding="utf-8") as output_loss, \
                         open(file=output_acc_path, mode="a", encoding="utf-8") as output_acc, \
                         open(file=output_val_loss_path, mode="a", encoding="utf-8") as output_val_loss, \
                         open(file=output_val_acc_path, mode="a", encoding="utf-8") as output_val_acc, \
                         open(file=output_modelo_final_path, mode="a", encoding="utf-8") as output_modelo_final:

                        cnn_model_name = feature_extractor_models_names[feature_extractor_models.index(feature_extractor_model)]
                        optimizer_name = optimizers_names[optimizers.index(optimizer)]

                        print("###############_" + str(count_variation) + "_of_" + str(total_variations) + "_###############\n")
                        output_all.write("###############_" + str(count_variation) + "_of_" + str(total_variations) + "_###############\n\n")
                        print("CNN: " + cnn_model_name)
                        output_all.write("CNN: " + cnn_model_name + "\n")
                        print("Transfer_Learning: " + ("True" if weight == "imagenet" else "False"))
                        output_all.write("Transfer_Learning: " + ("True" if weight == "imagenet" else "False") + "\n")
                        print("Trainable_Weights: " + str(trainable))
                        output_all.write("Trainable_Weights: " + str(trainable) + "\n")
                        print("Image_Augmentation: " + ("False" if image_augmentation is None else "True"))
                        output_all.write("Image_Augmentation: " + ("False" if image_augmentation is None else "True") + "\n")
                        print("Optimizer: " + optimizer_name + "\n")
                        output_all.write("Optimizer: " + optimizer_name + "\n")

                        model_variation_name = cnn_model_name + "_" + optimizer_name + ("_TL" if weight == "imagenet" else "") + ("_TR" if trainable else "") + ("_IA" if image_augmentation is not None else "")

                        start_time = time()
                        history = caption_model.fit(
                            train_dataset,
                            epochs=EPOCHS,
                            validation_data=valid_dataset,
                            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
                        )
                        end_time = time()

                        output_all.write("\nTraining: ")

                        loss = str(history.history['loss']).replace("[", "").replace("]", "").replace(",", "").replace(".", ",")
                        acc = str(history.history['acc']).replace("[", "").replace("]", "").replace(",", "").replace(".", ",")
                        val_loss = str(history.history['val_loss']).replace("[", "").replace("]", "").replace(",", "").replace(".", ",")
                        val_acc = str(history.history['val_acc']).replace("[", "").replace("]", "").replace(",", "").replace(".", ",")

                        output_all.write("\nloss: " + loss)
                        output_all.write("\nacc: " + acc)
                        output_all.write("\nval_loss: " + val_loss)
                        output_all.write("\nval_acc: " + val_acc + "\n")

                        output_loss.write(model_variation_name + " " + cnn_model_name + " " + optimizer_name + " " + ("True" if weight == "imagenet" else "False") + " " + str(trainable) + " " + ("True" if image_augmentation is not None else "False") + " " + loss + "\n")
                        output_acc.write(model_variation_name + " " + cnn_model_name + " " + optimizer_name + " " + ("True" if weight == "imagenet" else "False") + " " + str(trainable) + " " + ("True" if image_augmentation is not None else "False") + " " + acc + "\n")
                        output_val_loss.write(model_variation_name + " " + cnn_model_name + " " + optimizer_name + " " + ("True" if weight == "imagenet" else "False") + " " + str(trainable) + " " + ("True" if image_augmentation is not None else "False") + " " + val_loss + "\n")
                        output_val_acc.write(model_variation_name + " " + cnn_model_name + " " + optimizer_name + " " + ("True" if weight == "imagenet" else "False") + " " + str(trainable) + " " + ("True" if image_augmentation is not None else "False") + " " + val_acc + "\n")

                        duration_training = str(end_time - start_time).replace(".", ",")

                        ###########################
                        # Check sample predictions#
                        ###########################

                        # print("Predições: ")
                        output_all.write("\nPredictions: \n")
                        # Check predictions for all validation images
                        start_time = time()
                        for image_index in range(0, len(valid_images)):
                            image, predicted, label = generate_caption(image_index)
                            # print(image)
                            # print(predicted)
                            # print(label)
                            output_all.write(image + "\n")
                            output_all.write(predicted + "\n")
                            output_all.write(label + "\n")

                            predicted_exam = ''
                            predicted_body_part = ''
                            predicted_problem = ''

                            labeled_exam, labeled_body_part, labeled_problem = label.strip("Label:").split()
                            predicted_sequence = predicted.split()[1:]

                            if len(predicted_sequence) == 3:
                                predicted_exam, predicted_body_part, predicted_problem = predicted_sequence
                            elif len(predicted_sequence) == 2:
                                predicted_exam, predicted_body_part = predicted_sequence
                            elif len(predicted_sequence) == 1:
                                predicted_exam = predicted_sequence

                            # print("predicted_problem: " + predicted_problem)
                            # print("labeled_problem: " + labeled_problem)

                            if predicted_exam not in val_list:
                                predicted_exam = '[UNK]'

                            if predicted_body_part not in val_list:
                                predicted_body_part = '[UNK]'

                            if predicted_problem not in val_list:
                                predicted_problem = '[UNK]'

                            confusion_matrix_type_exam[val_list.index(labeled_exam)][val_list.index(predicted_exam)] += 1
                            confusion_matrix_body_part[val_list.index(labeled_body_part)][val_list.index(predicted_body_part)] += 1
                            confusion_matrix_problem_type[val_list.index(labeled_problem)][val_list.index(predicted_problem)] += 1


                        with open(file=output_confusion_matrix_type_exam_path, mode="a", encoding="utf-8") as output_confusion_matrix_type_exam, \
                             open(file=output_confusion_matrix_body_part_path, mode="a", encoding="utf-8") as output_confusion_matrix_body_part, \
                             open(file=output_confusion_matrix_problem_type_path, mode="a", encoding="utf-8") as output_confusion_matrix_problem_type:

                            output_confusion_matrix_type_exam.write(str(confusion_matrix_type_exam))
                            output_confusion_matrix_body_part.write(str(confusion_matrix_body_part))
                            output_confusion_matrix_problem_type.write(str(confusion_matrix_problem_type))

                        end_time = time()

                        duration_test = str(end_time - start_time).replace(".", ",")

                        final_loss = str(tf.keras.backend.get_value(caption_model.metrics[0].result())).replace(".", ",")
                        final_acc = str(tf.keras.backend.get_value(caption_model.metrics[1].result())).replace(".", ",")
                        print("\nFinal_Model: " + model_variation_name)
                        output_all.write("Final_Model: " + model_variation_name)
                        print("loss: " + final_loss)
                        print("acc: " + final_acc)
                        print("epochs: " + str(len(history.history['loss'])))
                        print("training_time(s): " + duration_training)
                        print("test_time(s): " + duration_test + "\n")
                        output_all.write("\nloss: " + final_loss)
                        output_all.write("\nacc: " + final_acc)
                        output_all.write("\nepochs: " + str(len(history.history['loss'])))
                        output_all.write("\ntraining_time(s): " + duration_training)
                        output_all.write("\ntest_time(s): " + duration_test + "\n\n")

                        output_modelo_final.write(model_variation_name + " " + cnn_model_name + " " + optimizer_name + " " + ("True" if weight == "imagenet" else "False") + " " + str(trainable) + " " + ("True" if image_augmentation is not None else "False") + " " + final_loss + " " + final_acc + " " + str(len(history.history['loss'])) + " " + duration_training + " " + duration_test + "\n")
