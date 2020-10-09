"""
Image segmentation task

This program loads manually labbelled wave image data and classify
each pixel in the image into "breaking" (1) or "no-breaking" (0).

The data needs to be in a folder which has sub-folders "images" and "masks"

For example:
```
└───data
    ├───images
    ├───masks
```

The neural net is a modified UNet architeture from here:

https://keras.io/examples/vision/oxford_pets_image_segmentation/

PROGRAM   : train_xception.py
POURPOSE  : segment wave breaking using a convnets
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V2.0      : 28/09/2020 [Caio Stringari]

"""

import os
import platform

import datetime

import random

import argparse

import numpy as np
import pandas as pd

from glob import glob

import collections
import math
import string

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import callbacks
# from tensorflow.keras import metrics

# TF addons
# import tensorflow_addons as tfa

# plot
import matplotlib.pyplot as plt


class Helper(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]
        x = np.zeros((batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = np.array(img) / 255.0
        y = np.zeros((batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size,
                           color_mode="grayscale")
            img = np.array(img)
            img = (img > 0).astype(int)  # ckeck here?
            y[j] = np.expand_dims(img, 2)  # why 2?
        return x, y

class SEBlock(Layer):
    """Implementation of Squeeze and Excitation network"""
    def __init__(self, name, filters, ratio, kernel_initializer,
                    activation=tf.nn.swish):
        super(SEBlock, self).__init__(name)

        self.squeeze = GlobalAveragePooling2D(name='se_squeeze')
        self.reduce = Conv2D(filters * ratio,
                                kernel_size=1,
                                activation=activation,
                                padding='same',
                                use_bias=True,
                                kernel_initializer=kernel_initializer,
                                name='se_reduce')

        self.expand = Conv2D(filters,
                                kernel_size=1,
                                activation='sigmoid',
                                padding='same',
                                use_bias=True,
                                kernel_initializer=kernel_initializer,
                                name='se_expand')

    def call(self, input):
        x = self.squeeze(input)
        channels = x.shape[-1]
        x = tf.reshape(x, (-1, 1, 1, channels))
        x = self.reduce(x)
        x = self.expand(x)
        return x

class MBConvBlock(Layer):
    def __init__(self, block_args, kernel_initializer, drop_rate=None,
                 name='', activation=tf.nn.swish):
        super(MBConvBlock, self).__init__(name=name)
        filters = block_args.input_filters * block_args.expand_ratio
        self.block_args = block_args
        if block_args.expand_ratio != 1:
            self.expand_conv = Conv2D(filters, 1, padding='same',
                                        use_bias=False,
                                        kernel_initializer=kernel_initializer,
                                        name='expand_cov')
            self.expand_bn = BatchNormalization(axis=3, name='expand_bn')
            self.expand_act = Activation(activation, name='expand_activation')

        self.depth_conv = DepthwiseConv2D(block_args.kernel_size,
                              strides=block_args.strides,
                              padding='same',
                              use_bias=False,
                              depthwise_initializer=kernel_initializer,
                              name='dwconv')
        self.depth_bn = BatchNormalization(axis=3, name='bn')
        self.depth_act = Activation(activation, name='activation')

        if block_args.se_ratio > 0:
            self.se_block = SEBlock('se', filters , block_args.se_ratio,
                                    kernel_initializer)

        self.out_conv = Conv2D(block_args.output_filters, 1,
                                padding='same',
                                use_bias=False,
                                kernel_initializer=kernel_initializer,
                                name='project')

        self.out_bn =  BatchNormalization(axis=3, name='project_bn')

    def call(self, input, training):
        x = input
        if self.block_args.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_act(x)
            enc = x

        x = self.depth_conv(x)
        x = self.depth_bn(x)
        x = self.depth_act(x)

        if self.block_args.se_ratio > 0:
            se = self.se_block(x)
            x = tf.math.multiply(x, se)

        x = self.out_conv(x)
        if self.block_args.final_bn:
          x = self.out_bn(x)

        if self.block_args.id_skip and \
              self.block_args.input_filters == self.block_args.output_filters:
            x = tf.math.add(x, input)

        return x, enc


# def get_model(img_size, num_classes):
#     """Define the model."""
#     inputs = keras.Input(shape=img_size + (3,))
#
#     # -- [First half of the network: downsampling inputs] ---
#
#     # entry block
#     x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation("relu")(x)
#
#     previous_block_activation = x  # Set aside residual
#
#     # blocks 1, 2, 3 are identical apart from the feature depth.
#     for filters in [64, 128, 256]:
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.SeparableConv2D(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
#
#         # Project residual
#         residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
#             previous_block_activation
#         )
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#
#     # --- [Second half of the network: upsampling inputs] ---
#
#     for filters in [256, 128, 64, 32]:
#         x = layers.Activation("relu")(x)
#         x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation("relu")(x)
#         x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.UpSampling2D(2)(x)
#
#         # Project residual
#         residual = layers.UpSampling2D(2)(previous_block_activation)
#         residual = layers.Conv2D(filters, 1, padding="same")(residual)
#         x = layers.add([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside next residual
#
#     # Add a per-pixel classification layer
#     outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
#
#     # Define the model
#     model = keras.Model(inputs, outputs)
#     return model


def display_mask(val_preds, i):
    """Display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return mask


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument("--data", "-i",
                        action="store",
                        dest="data",
                        required=True,
                        help="Input data path.",)
    # model
    parser.add_argument("--model", "-m",
                        action="store",
                        dest="model",
                        required=True,
                        help="Model name.",)

    # loggir
    parser.add_argument("--logdir", "-logdir",
                        action="store",
                        required=True,
                        dest="logdir",
                        help="Logging directory for Tensorboard.",)

    # random state seed for reproducibility
    parser.add_argument("--random-state", "-random-state",
                        action="store",
                        dest="random_state",
                        default=11,
                        required=False,
                        help="Random state.",)
    # validation size
    parser.add_argument("--test-size", "-testsize",
                        action="store",
                        dest="test_size",
                        default=0.2,
                        required=False,
                        help="validation size. Default is 0.1",)
    # learning rate
    parser.add_argument("--learning-rate", "-lr",
                        action="store",
                        dest="learning_rate",
                        default=10E-6,
                        required=False,
                        help="Learning rate. Default is 10E-6.",)
    # epochs
    parser.add_argument("--epochs", "-epochs",
                        action="store",
                        dest="epochs",
                        default=200,
                        required=False,
                        help="Learning rate. Default is 200.",)
    # Batch Size
    parser.add_argument("--batch-size", "-batch-size",
                        action="store",
                        dest="batch_size",
                        default=64,
                        required=False,
                        help="Batch size. Default is 64.",)

    parser.add_argument("--input-size", "-input-size",
                        nargs=2,
                        action="store",
                        dest="size",
                        default=[256, 256],
                        required=False,
                        help="Image input size. Default is 256x256.",)

    args = parser.parse_args()

    # --- I/O ---
    model_name = args.model
    img_size = (int(args.size[0]), int(args.size[1]))
    batch_size = int(args.batch_size)
    random_seed = int(args.random_state)
    epochs = int(args.epochs)
    logdir = args.logdir
    test_size = float(args.test_size)
    learning_rate = float(args.learning_rate)

    # --- Model Constants ---
    BlockArgs = collections.namedtuple('BlockArgs',
                                       ['kernel_size', 'num_repeat',
                                        'input_filters', 'output_filters',
                                        'expand_ratio', 'id_skip', 'strides',
                                        'se_ratio', 'final_bn'])

    BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

    CONV_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 2.0,
            'mode': 'fan_out',
            # EfficientNet actually uses an untruncated normal distribution for
            # initializing conv layers, but keras.initializers.VarianceScaling use
            # a truncated distribution.
            # We decided against a custom initializer for better serializability.
            'distribution': 'normal'
        }
    }

    DENSE_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 1. / 3.,
            'mode': 'fan_out',
            'distribution': 'uniform'
        }
    }

    DEFAULT_BLOCKS_ARGS = [
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                  expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25,
                  final_bn=True),
        BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25,
                  final_bn=True),
        BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25,
                  final_bn=True),
        BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25,
                  final_bn=True),
        BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                  expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25,
                  final_bn=True),
        BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                  expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25,
                  final_bn=True),
        BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                  expand_ratio=6, id_skip=False, strides=[1, 1], se_ratio=0.25,
                  final_bn=False)

    ]

    ENCODER_LAYERS = ['block2a', 'block3a', 'block4a',  'block6a']

    # --- define model metrics ---
    # metrics = [keras.metrics.BinaryAccuracy(name="Accuracy")]

    # --- Callbacks ---

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if platform.system().lower() == "windows":
        logdir = logdir + "\\" + model_name + "\\" + date
    else:
        logdir = logdir + "/" + model_name + "/" + date
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)

    tensorboard = callbacks.TensorBoard(log_dir=logdir,
                                        histogram_freq=1,
                                        profile_batch=1)

    if platform.system().lower() == "windows":
        checkpoint_path = logdir + "\\" + "best.h5"
    else:
        checkpoint_path = logdir + "/" + "best.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           monitor='val_loss',
                                           mode="min",
                                           verbose=1)
    if platform.system().lower() == "windows":
        pred_out = logdir + "\\" + "pred"
    else:
        pred_out = logdir + "/" + "pred"
    os.makedirs(pred_out, exist_ok=True)

    # --- Data ---

    # there is only 2 classes in this dataset, 0 or 1
    num_classes = 2

    # inputs
    input_dir = args.data + "/images/*.png"  # must be pngs
    target_dir = args.data + "/masks/*.png"  # must be pngs

    input_img_paths = sorted(glob(input_dir))
    target_img_paths = sorted(glob(target_dir))

    if not input_img_paths:
        raise IOError("Check your input images.")
    if not input_img_paths:
        raise IOError("Check your input masks.")

    # check your data
    print("\nNumber of samples = ", len(input_img_paths))
    print("\nFirst 10 images:")
    for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print("  ", input_path, "|", target_path)

    # --- Model ---

    # free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # # define the model
    # model = get_model(img_size, num_classes)
    # model.summary()
    #
    # # split img paths into a training and a validation set
    # val_samples = int(len(input_img_paths) * test_size)
    # random.Random(random_seed).shuffle(input_img_paths)
    # random.Random(random_seed).shuffle(target_img_paths)
    # train_input_img_paths = input_img_paths[:-val_samples]
    # train_target_img_paths = target_img_paths[:-val_samples]
    # val_input_img_paths = input_img_paths[-val_samples:]
    # val_target_img_paths = target_img_paths[-val_samples:]
    #
    # # instantiate data Sequences for each split
    # train_gen = Helper(batch_size, img_size, train_input_img_paths,
    #                    train_target_img_paths)
    # val_gen = Helper(batch_size, img_size, val_input_img_paths,
    #                  val_target_img_paths)
    #
    # # configure the model for training.
    # # we use the "sparse" version of categorical_crossentropy
    # # because our target data are integers.
    # optimizer = optimizers.Adam(learning_rate=learning_rate)
    # # loss = tfa.losses.giou_loss
    # # loss = ?
    # model.compile(optimizer=optimizer,
    #               loss="sparse_categorical_crossentropy",
    #               metrics=["accuracy"])
    #
    # # train the model, doing validation at the end of each epoch.
    # history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
    #                     callbacks=[tensorboard, checkpoint])
    # hist = pd.DataFrame(history.history)
    # hist["epoch"] = history.epoch
    #
    # # predict on the test data and save the outputs
    # val_gen = Helper(batch_size, img_size, val_input_img_paths,
    #                  val_target_img_paths)
    # val_preds = model.predict(val_gen)
    #
    # for i in range(val_preds.shape[0]):
    #     # load image
    #     img = load_img(val_input_img_paths[i])
    #     # load mask
    #     msk = load_img(val_target_img_paths[i])
    #     # process prediction
    #     prd = display_mask(val_preds, i)
    #     # plot
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6),
    #                                         sharex=True, sharey=True)
    #     ax1.imshow(img)
    #     ax2.imshow(msk)
    #     ax3.imshow(np.squeeze(prd))
    #     fig.tight_layout()
    #     plt.savefig(os.path.join(pred_out, str(i).zfill(6) + ".png"),
    #                 pad_inches=0.1, bbox_inches='tight')
    #     plt.close()
    #
    # #  --- Output ---
    # model.save("{}.h5".format(os.path.join(logdir, model_name)))
    # hist.to_csv("{}.cvs".format(os.path.join(logdir, model_name)))
