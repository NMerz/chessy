import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from itertools import chain
from functools import partial
import pathlib
import os
import PIL.Image as Image
from typing import Dict

keras.config.disable_traceback_filtering()

size = (256, 64)


def file_to_numpy(to_convert: pathlib.Path) -> np.array:
    img = Image.open(to_convert).convert("RGBA")
    resized = img.resize(size).convert("L")
    return np.expand_dims(np.array(resized), axis=2)  # add channel dimension


def get_char_key(char_key_file: pathlib.Path):
    char_key = {}
    with open(char_key_file) as ckf:
        char_key = {char.strip(): i for i, char in enumerate(ckf.readlines())}
    return char_key


def invert_map(to_invert):
    new_map = {}
    for k, v in to_invert.items():
        new_map[v] = k
    return new_map


def encode_words(
    char_key: Dict[str, int],
    file_word_mapping: [str, str],
    max_length: int,
) -> Dict[str, any]:

    label_map = {}

    for filename, train_str in file_word_mapping.items():
        tk = [None] * max_length
        for c_i in range(0, max_length):
            if len(train_str) <= c_i:
                tk[c_i] = 0
                continue
            char_i = char_key[train_str[c_i]]
            tk[c_i] = char_i
        label_map[filename] = tk
    return label_map


def snoopy_loss(y_true, y_pred, **kwargs):
    y_true = tf.keras.ops.cast(y_true, dtype="int32")
    print("Snoopy")
    print(y_true)
    print(y_pred)
    # label_length = ([label_length] * batch_size,)
    label_length = tf.math.count_nonzero(y_true, axis=-1)
    print(label_length)
    return tf.nn.ctc_loss(y_true, y_pred, **kwargs, label_length=label_length)


def train_model(
    train_inputs, train_labels, label_length, char_num, save_name, char_key
):
    batch_size = 1
    train_inputs = train_inputs[: len(train_inputs) // batch_size * batch_size]
    train_labels = train_labels[: len(train_labels) // batch_size * batch_size]
    train_ds_labeled = tf.data.Dataset.zip(
        tf.data.Dataset.from_tensor_slices(train_inputs),
        tf.data.Dataset.from_tensor_slices(train_labels),
    ).batch(batch_size)

    # train_ds_labeled = train_ds.map(lambda v, l: (v, label_map[l]))
    """
    tf.data.Dataset.zip(
        (train_ds, tf.data.Dataset.from_tensor_slices(oh_training_keys_batch))
    )
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(64, 256, 1)),
            # tf.keras.layers.Input(shape=(1, 1, 1)),
            tf.keras.layers.Conv2D(
                64, 5, strides=2, padding="same", activation="relu"
            ),  # Kernel size is irrelevant to output dimension given padding. 5 seems common
            tf.keras.layers.Conv2D(
                128, 5, strides=2, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2D(
                256, 5, strides=1, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2D(
                512, 5, strides=(2, 1), padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2D(
                512, 5, strides=1, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv2D(
                512, 5, strides=(2, 1), padding="same", activation="relu"
            ),
            # tf.keras.layers.Permute([3, 1, 2]),
            # tf.keras.layers.Reshape((512, -1)),
            tf.keras.layers.Permute([2, 1, 3]),
            tf.keras.layers.Reshape((64, -1)),
            # tf.keras.layers.Bidirectional(
            # tf.keras.layers.LSTM(256, return_sequences=True)
            # ),
            # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                label_length * char_num,
                activation=partial(tf.nn.leaky_relu, alpha=0.01),
            ),
            tf.keras.layers.Reshape([label_length, char_num]),
        ]
    )

    try:
        model.load_weights(save_name)
        pass
    except:
        pass
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        # optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        # loss=tf.keras.losses.CategoricalCrossentropy,
        # loss=tf.keras.losses.CTC,
        loss=partial(
            snoopy_loss,
            logit_length=[char_num] * batch_size,
            blank_index=0,
            logits_time_major=False,
        ),
        metrics=["accuracy"],
    )

    for layer in model.layers:
        print(layer.output.shape)
    # for layer in model.layers:
    # print(layer.name)
    # print(layer.get_weights())

    # print(model.predict(tf.data.Dataset.from_tensor_slices(train_inputs).batch(2)))

    """
    with tf.device("/cpu:0"):
        model.fit(train_ds_labeled, epochs=100)
    """

    # for layer in model.layers:
    # print(layer.name)
    # print(layer.get_weights())
    # print(train_inputs)
    # with tf.device("/cpu:0"):
    print(train_labels)
    print(
        tf.math.argmax(
            model.predict(tf.data.Dataset.from_tensor_slices(train_inputs).batch(2)), 2
        )
    )

    char_key = invert_map(char_key)

    def get_char(x):
        y = tf.keras.backend.get_value(x)
        # print(y)
        # print(char_key.get(y, ""))
        return char_key.get(y, "")

    def pad_2d(to_pad, pad_to):
        current_dims = tf.shape(to_pad)[0]
        return tf.pad(to_pad, [[0, pad_to - current_dims]], "CONSTANT")

    predictions = (
        tf.math.argmax(
            model.predict(tf.data.Dataset.from_tensor_slices(train_inputs).batch(2)),
            2,
        ),
    )
    print(predictions)

    def apply_mask(to_mask):
        # print("masking")
        to_mask = to_mask[0]
        to_mask = tf.squeeze(to_mask, [0])
        # print(to_mask)
        # Mask duplicates from https://stackoverflow.com/a/51262258
        to_mask_shifted = tf.concat((tf.slice(to_mask, [1], [-1]), [0]), axis=0)
        # print(to_mask_shifted, flush=True)
        shifted_mask = tf.not_equal(to_mask - to_mask_shifted, 0)
        duplicate_mask = tf.concat(([True], shifted_mask[:-1]), axis=0)
        to_mask = tf.boolean_mask(to_mask, duplicate_mask)
        # print(to_mask)
        return tf.boolean_mask(to_mask, tf.not_equal(to_mask, 0))

    print(
        tf.map_fn(
            lambda single_pred: tf.strings.join(
                tf.strings.join(
                    tf.map_fn(
                        lambda x: tf.py_function(get_char, [x], [tf.string]),
                        apply_mask(single_pred),
                        # pad_2d(tf.boolean_mask(single_pred, tf.greater(single_pred, 0)), 30),
                        infer_shape=False,
                        fn_output_signature=[tf.string],
                    ),
                )
            ),
            [predictions],
            infer_shape=False,
            fn_output_signature=tf.string,
        ),
    )

    # model.save_weights(save_name)


with open("/Users/nmerz/Downloads/ascii/words.txt") as word_key:
    key_lines = [
        line.strip().split(" ") for line in word_key.readlines() if line[0] != "#"
    ]

chars_per_word = 12

file_word_mapping = {}
for line in key_lines:
    if len(line) < 9:
        continue
    if line[1] != "ok":
        continue
    file_word_mapping[line[0]] = line[8]

generic_data_dir = pathlib.Path("/Users/nmerz/Documents/chessy/small_generic")
# generic_data_dir = pathlib.Path("/Users/nmerz/Downloads/words")
generic_char_key = get_char_key(
    pathlib.Path("/Users/nmerz/Downloads/words/_char_table.txt")
)
generic_pre = encode_words(
    generic_char_key,
    {
        k: file_word_mapping.get(k.removesuffix(".png"), "")
        for k in chain.from_iterable(
            [dir_contents[2] for dir_contents in os.walk(generic_data_dir)]
        )
        if k != ".DS_Store" and k.removesuffix(".png") in file_word_mapping.keys()
    },
    31,
)

image_list = []
word_list = []
for generic_word_image in generic_data_dir.glob("*/*/*"):
    try:
        if generic_word_image.name in generic_pre.keys():
            image_list.append(file_to_numpy(generic_word_image))
            word_list.append(generic_pre[generic_word_image.name])
    except:
        pass

# word_list = [[1, 2], [3, 2]]
# print(word_list)

# train_model([[[1]], [[2]]], word_list, 2, 4, "generic_train_weights.weights.h5")
train_model(
    image_list[:5],
    word_list[:5],
    31,
    80,
    "generic_train_weights.weights.h5",
    generic_char_key,
)
# train_model(image_list, word_list, 31, 80, "generic_train_weights.weights.h5")

exit(0)

data_dir = pathlib.Path(
    # "/Users/nmerz/Downloads/HCS Dataset December 2021/extracted move boxes"
    "/Users/nmerz/Documents/chessy/small_test"
).with_suffix("")

img_height = 64
img_width = 256


with open(
    "/Users/nmerz/Downloads/HCS Dataset December 2021/extracted move boxes/training_tags.txt"
) as tags:
    lines = [line.strip() for line in tags.readlines()]

training_tags = {line.split(" ")[0]: line.split(" ")[1] for line in lines}

label_map = encode_words(
    get_char_key(
        pathlib.Path(
            "/Users/nmerz/Downloads/HCS Dataset December 2021/extracted move boxes/char_table.txt"
        )
    ),
    {
        k: training_tags.get(k, "")
        for k in list(os.walk(data_dir))[0][2]
        if k != ".DS_Store"
    },
    chars_per_word,
)

"""
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    labels=list(range(len(training_key.keys()))),
    color_mode="grayscale",
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=1,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    labels=list(range(len(training_key.keys()))),
    subset="validation",
    color_mode="grayscale",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=1,
)
"""

size = (256, 64)
input_images = []
ordered_labels = []
for f in data_dir.glob("*"):
    input_images.append(file_to_numpy(f))
    ordered_labels.append(label_map[str(f.name)])

train_model(
    input_images, ordered_labels, chars_per_word, 30, "train_weights.weights.h5"
)
