"""
Segment wave breaking pixels with UNet-line conv-nets.

This program loads manually labbelled wave image data and classify
each pixel in the image into "breaking" (1) or "no-breaking" (0).

The data needs to be organized as follows:

For example:
```
└───train or test or valid
    ├───images
        ├───data
               ├───img1.png
               ├───img2.png
               ...
    ├───masks
        ├───data
               ├───img1.png
               ├───img2.png
               ...
```

The neural nets are modified UNets from:
https://keras.io/examples/vision/oxford_pets_image_segmentation/
https://www.tensorflow.org/tutorials/images/segmentation

PROGRAM   : train.py
POURPOSE  : segment wave breaking using a convnets
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V2.0      : 09/10/2020 [Caio Stringari]
"""

import os
import platform

import datetime

import argparse

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# TF addons
# import tensorflow_addons as tfa

# plot
import matplotlib.pyplot as plt


def xception(img_size, num_classes):
    """Define the model."""
    inputs = keras.Input(shape=img_size + (3,))

    # -- [First half of the network: downsampling inputs] ---

    # entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # --- [Second half of the network: upsampling inputs] ---

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(
        num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def mobilenet(img_size, num_classes, model_path="mobilenet.h5"):
    """Define the model."""
    # Use mobile net
    # base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3],
    #                                                include_top=False)
    base_model = load_model(model_path)

    # use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project']      # 4x4

    layers = [base_model.get_layer(name).output for name in layer_names]

    # create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    # create the upstack
    up_stack = [pix2pix.upsample(512, 3),  # 4x4 -> 8x8
                pix2pix.upsample(256, 3),  # 8x8 -> 16x16
                pix2pix.upsample(128, 3),  # 16x16 -> 32x32
                pix2pix.upsample(64, 3)]  # 32x32 -> 64x64

    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(num_classes, 3, strides=2,
                                           padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def display_mask(val_preds):
    """Display a model's prediction."""
    mask = np.argmax(val_preds, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def image_mask_generator(image_data_generator, mask_data_generator):
    """Yield a generator."""
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img[0], mask[0][:, :, :, 0])


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

    parser.add_argument("--backbone", "-b",
                        action="store",
                        dest="backbone",
                        default="xception",
                        required=False,
                        help="Model backbone. Default is xception.",)

    parser.add_argument("--pre-trained",
                        action="store",
                        dest="pre_trained",
                        default=None,
                        required=False,
                        help="Model pre-trained model for supported backbones .",)

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
    backbone = args.backbone
    pre_trained = args.pre_trained
    model_name = args.model
    img_size = (int(args.size[0]), int(args.size[1]))
    batch_size = int(args.batch_size)
    random_seed = int(args.random_state)
    epochs = int(args.epochs)
    logdir = args.logdir
    test_size = float(args.test_size)
    learning_rate = float(args.learning_rate)

    # there are only 2 classes in this dataset, 0 or 1
    num_classes = 2

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

    # --- Data Augmentation ----

    # train generators - they need to be identical!
    image_train_generator = ImageDataGenerator(
        zoom_range=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1. / 255.).flow_from_directory(args.data + "/train/images",
                                               batch_size=batch_size,
                                               target_size=img_size,
                                               seed=random_seed)
    mask_train_generator = ImageDataGenerator(
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1. / 255.).flow_from_directory(args.data + "/train/masks",
                                               batch_size=batch_size,
                                               target_size=img_size,
                                               seed=random_seed)

    # test/valid generators - they need to be identical!
    # note that no augmentation is really done to this data, only a resize
    image_valid_generator = ImageDataGenerator(
        rescale=1. / 255.).flow_from_directory(args.data + "/valid/images",
                                               batch_size=batch_size,
                                               target_size=img_size,
                                               seed=random_seed)

    mask_valid_generator = ImageDataGenerator(
        rescale=1. / 255.).flow_from_directory(args.data + "/valid/masks",
                                               batch_size=batch_size,
                                               target_size=img_size,
                                               seed=random_seed)
    image_test_generator = ImageDataGenerator(
        rescale=1. / 255.).flow_from_directory(args.data + "/test/images",
                                               batch_size=1,
                                               target_size=img_size,
                                               seed=random_seed)

    mask_test_generator = ImageDataGenerator(
        rescale=1. / 255.).flow_from_directory(args.data + "/test/masks",
                                               batch_size=1,
                                               target_size=img_size,
                                               seed=random_seed)

    train_generator = image_mask_generator(image_train_generator, mask_train_generator)
    valid_generator = image_mask_generator(image_valid_generator, mask_valid_generator)
    test_generator = image_mask_generator(image_test_generator, mask_test_generator)
    train_size = image_train_generator.n
    valid_size = image_valid_generator.n
    # test_size = image_valid_generator.n

    # --- Model ---

    # free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # define the model
    if backbone.lower() == "xception":
        model = xception(img_size, num_classes)
    elif backbone.lower() == "mobilenet":
        if not pre_trained:
            raise ValueError("Pre-trained model is required with {}.".format(backbone))
        model = mobilenet(img_size, num_classes, pre_trained)
    else:
        raise NotImplementedError("Backbone {} is not implemented".format(backbone))
    model.summary()

    # configure the model for training.
    # we use the "sparse" version of categorical_crossentropy
    # because our target data are integers.
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # train the model, doing validation at the end of each epoch.
    history = model.fit(train_generator,
                        epochs=epochs,
                        steps_per_epoch=train_size // batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_size // batch_size,
                        callbacks=[tensorboard, checkpoint])
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

# predict on the test data and save the outputs
for i in range(image_test_generator.n):
    img, msk = next(test_generator)
    val_preds = model.predict(img)
    # process prediction
    prd = display_mask(val_preds)
    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6),
                                        sharex=True, sharey=True)
    ax1.imshow(np.squeeze(img))
    ax2.imshow(np.squeeze(msk))
    ax3.imshow(np.squeeze(prd))
    fig.tight_layout()
    plt.savefig(os.path.join(pred_out, str(i).zfill(6) + ".png"),
                pad_inches=0.1, bbox_inches='tight')
    plt.close()

    #  --- Output ---
    model.save("{}.h5".format(os.path.join(logdir, model_name)))
    hist.to_csv("{}.cvs".format(os.path.join(logdir, model_name)))
