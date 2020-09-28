"""
Active Wave Breaking Classifier

This program loads manually labbelled wave image data and classify
the images into "breaking" (1) or "no-breaking" (0).

The data needs to be in a folder which has sub-folders "0" and "1"

For example:
```
└───Data
    ├───0
    ├───1
```

There are 5 "backbones" implemented:
    "VGG16", "ResNet50V2", "InceptionResNetV2", MobileNet and EfficientNet

Note that the weights from these pre-trained models will be reset and
updated from the scratch here.

These models have no knowledge of the present data and, consequently,
transfered learning does not work well.

PROGRAM   : train.py
POURPOSE  : classify wave breaking using a convnets
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V2.0      : 14/05/2020 [Caio Stringari]

"""

# Imports

# run this cell to use EfficientNet
# !pip install -U --pre efficientnet
# import efficientnet.tfkeras as efn
import os

import argparse

import numpy as np

import datetime

import pandas as pd

import pathlib

import platform

# from tensorflow import summary
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.applications import (VGG16, ResNet50V2,
                                           InceptionResNetV2, MobileNetV2)
from tensorflow.keras.layers import (Dense, Dropout, Flatten)
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from sklearn.utils import class_weight

try:
    import efficientnet.tfkeras as efn
except Exception:
    pass
    print(ImportError("\nWarning: run pip install -U --pre efficientnet"))


if __name__ == '__main__':

    print("\nClassifiying wave breaking data, please wait...\n")

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
    # backbone
    parser.add_argument("--backbone",
                        action="store",
                        default="VGG16",
                        dest="backbone",
                        required=False,
                        help="Which backbone to use.",)
    # loggir
    parser.add_argument("--logdir",
                        action="store",
                        required=True,
                        help="Logging directory for Tensorboard.",)
    # random state seed for reproducibility
    parser.add_argument("--random-state", "-random-state",
                        action="store",
                        dest="random_state",
                        default=11,
                        required=False,
                        help="Random state.",)
    # validation size
    parser.add_argument("--validation-size", "-valsize",
                        action="store",
                        dest="val_size",
                        default=0.2,
                        required=False,
                        help="validation size. Default is 0.2",)
    # learning rate
    parser.add_argument("--learning-rate", "-lr",
                        action="store",
                        dest="learning_rate",
                        default=0.00001,
                        required=False,
                        help="Learning rate. Default is 0.00001.",)
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
    # Dropout
    parser.add_argument("--dropout", "-dropout",
                        action="store",
                        dest="dropout",
                        default=0.5,
                        required=False,
                        help="Dropout. Default is 0.5.",)
    # image size
    parser.add_argument("--input-size", "-input-size",
                        nargs=2,
                        action="store",
                        dest="imgsize",
                        default=[256, 256],
                        required=False,
                        help="Image input size. Default is 256x256.",)

    # parse the constants
    args = parser.parse_args()
    RANDOM_STATE = int(args.random_state)
    TEST_SIZE = float(args.val_size)
    LEARNING_RATE = float(args.learning_rate)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    DROPOUT = float(args.dropout)
    IMGSIZE = (int(args.imgsize[0]), int(args.imgsize[1]))
    LOGDIR = args.logdir
    DATA = args.data
    BACKBONE = args.backbone
    NAME = args.model

    # --- define model metrics ---
    METRICS = [metrics.TruePositives(name="True_Positives"),
               metrics.FalsePositives(name="False_Positives"),
               metrics.TrueNegatives(name="True_Negatives"),
               metrics.FalseNegatives(name="False_Negatives"),
               metrics.BinaryAccuracy(name="Binary_Accuracy"),
               metrics.Precision(name="Precision"),
               metrics.Recall(name="Recall"),
               metrics.AUC(name="AUC")]

    # --- tensorflow calbacks ---
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if platform.system().lower() == "windows":
        LOGDIR = LOGDIR + "\\" + NAME + "\\" + date
    else:
        LOGDIR = LOGDIR + "/" + NAME + "/" + date
    if not os.path.isdir(LOGDIR):
        os.makedirs(LOGDIR, exist_ok=True)

    tensorboard = callbacks.TensorBoard(log_dir=LOGDIR,
                                        histogram_freq=1,
                                        profile_batch=1)

    if platform.system().lower() == "windows":
        checkpoint_path = LOGDIR + "\\" + "best.h5"
    else:
        checkpoint_path = LOGDIR + "/" + "best.h5"
    checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           monitor='val_loss',
                                           mode="min",
                                           verbose=1)

    # --- data augumentation ---

    # Because the dataset has relatively few samples (20k),
    # we need some data augumentation so that our model trains better.

    # The augumentations steps are:
    # 1. Rescale in the range [0, 1]
    # 2. Rotate the data in 20 degrees angles
    # 3. ~~Shift the data in the horizontal and vertical orientations~~
    # 4. Flip up-down and left-right
    # 5. Zoom in and out by 20%

    # Keras generators will take care of all this  for us.

    # --- data input ---
    data_dir = pathlib.Path(DATA)
    image_count = len(list(data_dir.glob('*/*')))

    class_names = np.array([item.name for item in data_dir.glob('*')])

    try:
        nclasses = len(class_names)
        print("  Found image data, proceeding.\n")
        print("   - Classes are {}".format(class_names))
    except Exception:
        raise IOError("Check your data!")

    img_height = IMGSIZE[0]  # image height for all images
    img_width = IMGSIZE[1]  # image width for all images

    # tells the Generator when to stop
    steps_per_epoch = np.ceil(image_count / BATCH_SIZE)

    datagen = ImageDataGenerator(zoom_range=0.2,
                                 rotation_range=20,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 rescale=1. / 255.,
                                 validation_split=TEST_SIZE)

    print("\n    Fitting the training data generator:\n")
    data_gen_train = datagen.flow_from_directory(
        directory=str(data_dir), batch_size=BATCH_SIZE, shuffle=True,
        target_size=(img_height, img_width), classes=["0", "1"],
        subset='training', class_mode="binary")

    print("\n    Fitting the validation data generator:\n")
    data_gen_valid = datagen.flow_from_directory(
        directory=str(data_dir), batch_size=BATCH_SIZE, shuffle=True,
        target_size=(img_height, img_width), classes=["0", "1"],
        subset='validation', class_mode="binary")

    train_size = data_gen_train.n
    val_size = data_gen_valid.n

    # compute class weights
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(data_gen_train.classes),
        data_gen_train.classes)

    print("\nClass weights are:")
    weights = {0: class_weights[0], 1: class_weights[1]}
    print(weights)

    # --- Model Definition ----

    # Please use "VGG16", "ResNetV250", or "InceptionResNetV2", "MobileNetV2".
    # All other architetures must be adapted.

    # Note that I am using Flatten() instead of
    # Global Average Pooling(). The model seems to learn better using
    # Flatten().

    print("\n - Training using {} as backbone".format(BACKBONE))

    # define backbone
    if BACKBONE.lower() == "VGG16".lower():
        base_model = VGG16(input_shape=(img_height, img_width, 3),
                           include_top=False,
                           weights=None)
    elif BACKBONE.lower() == "ResNet50V2".lower():
        base_model = ResNet50V2(input_shape=(img_height, img_width, 3),
                                include_top=False,
                                weights=None)
    elif BACKBONE.lower() == "InceptionResNetV2".lower():
        base_model = InceptionResNetV2(input_shape=(img_height, img_width, 3),
                                       include_top=False,
                                       weights=None)
    elif BACKBONE.lower() == "EfficientNet".lower():
        base_model = efn.EfficientNetB5(input_shape=(img_height, img_width, 3),
                                        weights=None,
                                        include_top=False,)
    elif BACKBONE.lower() == "MobileNetV2".lower():
        base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                                 include_top=False,
                                 weights=None)
    else:
        raise NotImplementedError("Unknown backbone \'{}\' ".format())
    base_model.trainable = True

    # define top layers
    x = base_model.output
    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)  # this dropout here also seems to help
    x = Dense(2048)(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(2048)(x)
    x = Dropout(DROPOUT)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print("    + Model has been defined")
    # model.summary()

    # compile the model
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = BinaryCrossentropy(from_logits=False)
    model.compile(optimizer=optimizer,
                  loss=loss,  # try categorical_hinge later
                  metrics=METRICS)
    print("    + Model has been compiled")

    #  --- Training Loop ---

    history = model.fit(data_gen_train,
                        class_weight=weights,
                        epochs=EPOCHS,
                        steps_per_epoch=train_size // BATCH_SIZE,
                        verbose=1,
                        validation_data=data_gen_valid,
                        validation_steps=val_size // BATCH_SIZE,
                        callbacks=[tensorboard, checkpoint])
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    #  --- Output ---
    model.save("{}.h5".format(os.path.join(LOGDIR, NAME)))
    hist.to_csv("{}.cvs".format(os.path.join(LOGDIR, NAME)))
