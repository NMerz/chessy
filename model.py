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
from PIL import Image, ImageOps
from typing import Dict
from affine import Affine

keras.config.disable_traceback_filtering()

img_height = 64
img_width = 256
# size = (256, 64)
size = (img_width, img_height)


def img_to_numpy(img) -> np.array:
    return np.expand_dims(
        np.divide(np.array(img), 255.0), axis=2
    )  # add channel dimension


def file_to_numpy_arrs(to_convert: pathlib.Path, to_augment=False) -> [np.array]:
    img = Image.open(to_convert).convert("RGBA")
    resized = img.resize(size).convert("L")
    if not to_augment:
        return [img_to_numpy(resized)]
    mutated_imgs = [resized]
    # print("pix0", resized.load()[0, img_height // 2])
    for _ in range(9):
        mutation_type = random.randint(1, 3)
        if mutation_type == 1:
            mutated_imgs.append(
                resized.rotate(
                    random.random() * 20 - 10,
                    fillcolor=resized.load()[0, img_height // 2],
                )
            )  # +- 10 degrees
        if mutation_type == 2:
            mutated_width = int(img_width * (1 + random.random() * 0.4 - 0.2))
            diff = mutated_width - img_width
            stretch_img = resized.resize((mutated_width, img_height))
            if diff > 0:
                mutated_imgs.append(
                    stretch_img.crop((diff / 2, 0, diff / 2 + img_width, img_height))
                )
            else:
                mutated_imgs.append(
                    ImageOps.expand(
                        stretch_img,
                        border=-diff // 2,
                        fill=resized.load()[0, img_height // 2],
                    )
                    # .crop((0, diff / 2, img_width, diff / 2 + img_height))
                )
                # print(stretch_img)
                # print(mutated_imgs[-1].size)
        if mutation_type == 3:
            shear = random.random() * 30 - 15
            mutated_imgs.append(
                resized.transform(
                    size,
                    Image.AFFINE,
                    Affine.shear(shear, 0)[:6],
                    fillcolor=resized.load()[0, img_height // 2],
                )
            )

    to_return = []
    for img in mutated_imgs:
        to_return.append(img_to_numpy(img.resize(size)))
    return to_return


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
    # print("Snoopy")
    # print(y_true)
    # print(y_pred)
    # label_length = ([label_length] * batch_size,)
    label_length = tf.math.count_nonzero(y_true, axis=-1)
    # print(label_length)
    return tf.nn.ctc_loss(
        y_true,
        y_pred,
        **kwargs,
        # label_length=[1]
        label_length=label_length
    )


def train_model(
    train_inputs, train_labels, label_length, char_num, save_name, char_key
):
    batch_size = 32
    # train_inputs = train_inputs[:5120]
    # train_labels = train_labels[:5120]
    train_inputs = train_inputs[: len(train_inputs) // batch_size * batch_size]
    train_labels = train_labels[: len(train_labels) // batch_size * batch_size]
    train_ds_labeled = (
        tf.data.Dataset.zip(
            tf.data.Dataset.from_tensor_slices(train_inputs),
            tf.data.Dataset.from_tensor_slices(train_labels),
        )
        .shuffle(buffer_size=1000)
        .batch(batch_size)
    )

    # train_ds_labeled = train_ds.map(lambda v, l: (v, label_map[l]))
    """
    tf.data.Dataset.zip(
        (train_ds, tf.data.Dataset.from_tensor_slices(oh_training_keys_batch))
    )
    """

    model = tf.keras.Sequential(
        [
            # tf.keras.layers.Input(shape=(64, 256, 1)),
            tf.keras.layers.Input(shape=(img_height, img_width, 1)),
            # tf.keras.layers.Input(shape=(1, 1, 1)),
            tf.keras.layers.Conv2D(
                16,
                3,
                strides=2,
                padding="same",
                activation=partial(tf.nn.leaky_relu, alpha=0.05),
            ),  # Kernel size is irrelevant to output dimension given padding. 5 seems common
            tf.keras.layers.Conv2D(
                32,
                3,
                strides=2,
                padding="same",
                activation=partial(tf.nn.leaky_relu, alpha=0.05),
            ),
            tf.keras.layers.Conv2D(
                64,
                1,
                strides=1,
                padding="same",
                activation=partial(tf.nn.leaky_relu, alpha=0.05),
            ),
            tf.keras.layers.Conv2D(
                64,
                3,
                strides=(2, 1),
                padding="same",
                activation=partial(tf.nn.leaky_relu, alpha=0.05),
            ),
            # tf.keras.layers.Conv2D(
            # 64,
            # 3,
            # strides=1,
            # padding="same",
            # activation=partial(tf.nn.leaky_relu, alpha=0.05),
            # ),
            # tf.keras.layers.Conv2D(
            # 64,
            # 3,
            # strides=(2, 1),
            # padding="same",
            # activation=partial(tf.nn.leaky_relu, alpha=0.05),
            # ),
            # tf.keras.layers.Permute([3, 1, 2]),
            # tf.keras.layers.Reshape((512, -1)),
            tf.keras.layers.Permute([2, 1, 3]),
            tf.keras.layers.Reshape((64, -1)),
            # tf.keras.layers.Bidirectional(
            # tf.keras.layers.LSTM(64, return_sequences=True)
            # ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                label_length * char_num,
                activation=partial(tf.nn.leaky_relu, alpha=0.05),
                # kernel_initializer=tf.keras.initializers.GlorotNormal,
            ),
            tf.keras.layers.Reshape([label_length, char_num]),
        ]
    )

    try:
        # model.load_weights(save_name, skip_mismatch=True)
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
        # run_eagerly=True,
    )

    for layer in model.layers:
        print(layer.output.shape)
    # for layer in model.layers:
    # print(layer.name)
    # print(layer.get_weights())

    # print(model.predict(tf.data.Dataset.from_tensor_slices(train_inputs).batch(2)))

    # print(train_labels)
    # print(train_inputs)
    with tf.device("/cpu:0"):
        model.fit(train_ds_labeled, epochs=50)

    model.save_weights(save_name + "_trained.weights.h5")

    # for layer in model.layers:
    # print(layer.name)
    # print(layer.get_weights())
    # print(train_inputs)
    # print(train_labels)
    # with tf.device("/cpu:0"):
    # print(train_labels)
    pred_dataset = tf.data.Dataset.from_tensor_slices(train_inputs).batch(1)
    # print(train_inputs)
    # print(pred_dataset)
    # print(model.predict(pred_dataset))

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
            model.predict(pred_dataset),
            2,
        ),
    )
    # print(predictions)

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

    # print(
    # [
    # "".join([get_char(x) for x in single_label if x != 0])
    # for single_label in train_labels
    # ]
    # )

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

# generic_data_dir = pathlib.Path("/Users/nmerz/Documents/chessy/small_generic")
generic_data_dir = pathlib.Path("/Users/nmerz/Downloads/words")
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
    30,
)

image_list = []
word_list = []
for generic_word_image in generic_data_dir.glob("*/*/*"):
    try:
        if generic_word_image.name in generic_pre.keys():
            image_list.append(file_to_numpy_arrs(generic_word_image)[0])
            word_list.append(generic_pre[generic_word_image.name])
    except:
        pass

# word_list = [[1, 2], [3, 2]]
# print(word_list)

# train_model([[[1]], [[2]]], word_list, 2, 4, "generic_train_weights.weights.h5")

"""
train_model(
    image_list[:12],
    word_list[:12],
    30,
    80,
    "generic_train_weights.weights.h5",
    generic_char_key,
)
exit(0)

train_model(
    image_list,
    word_list,
    30,
    80,
    "generic_train_weights_full_slim.weights.h5",
    generic_char_key,
)

exit(0)
"""

data_dir = pathlib.Path(
    "/Users/nmerz/Downloads/HCS Dataset December 2021/extracted move boxes"
    # "/Users/nmerz/Documents/chessy/small_test"
).with_suffix("")


with open(
    "/Users/nmerz/Downloads/HCS Dataset December 2021/extracted move boxes/training_tags.txt"
) as tags:
    lines = [line.strip() for line in tags.readlines()]

training_tags = {line.split(" ")[0]: line.split(" ")[1] for line in lines}

chess_char_key = get_char_key(
    pathlib.Path(
        "/Users/nmerz/Downloads/HCS Dataset December 2021/extracted move boxes/char_table.txt"
    )
)
label_map = encode_words(
    chess_char_key,
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

input_images = []
ordered_labels = []
for f in data_dir.glob("*"):
    # print(f.name)
    if f.name == ".DS_Store":
        continue
    if f.suffix != ".png":
        continue
    for img in file_to_numpy_arrs(f, to_augment=True):
        input_images.append(img)
        ordered_labels.append(label_map[str(f.name)])

train_model(
    input_images,
    ordered_labels,
    chars_per_word,
    30,
    "generic_train_weights_full_slim.weights.h5",
    # "generic_train_weights_full.weights.h5",
    # "generic_train_weights_full.weights.h5new.weights.h5",
    # "generic_train_weights_full.weights.h5new.weights.h5nolstm.weights.h5",
    chess_char_key,
)
exit(0)
train_model(
    input_images, ordered_labels, chars_per_word, 30, "train_weights.weights.h5"
)
