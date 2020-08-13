"""
This program loads labbelled wave image data and update the weights of the
nueral net

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

PROGRAM   : update.py
POURPOSE  : classify wave breaking using a convnets
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V1.0      : 12/08/2020 [Caio Stringari]

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
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


if __name__ == '__main__':

    print("\nUpdating model, please wait...\n")

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
                        help="Pre-trained model.",)

    parser.add_argument("--new-model", "-o",
                        action="store",
                        dest="newmodel",
                        required=True,
                        help="Updated model.",)

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

    # parse the constants
    args = parser.parse_args()
    RANDOM_STATE = int(args.random_state)
    TEST_SIZE = float(args.val_size)
    LEARNING_RATE = float(args.learning_rate)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    LOGDIR = args.logdir
    DATA = args.data
    NAME = args.model
    NEWNAME = args.newmodel

    # ---- load the pre-trained model ----
    model = tf.keras.models.load_model(NAME)

    # --- tensorflow calbacks ---
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if platform.system().lower() == "windows":
        LOGDIR = LOGDIR + "\\" + NEWNAME + "\\" + date
    else:
        LOGDIR = LOGDIR + "/" + NEWNAME + "/" + date
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

    # The augumentations steps are:
    # 1. Rescale in the range [0, 1]
    # 3. Rotate the data in 20 degrees angles
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

    inp_shape = model.input_shape
    img_height = inp_shape[1]  # image height for all images
    img_width = inp_shape[2]  # image width for all images

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

    #  --- Training Loop ---
    history = model.fit(data_gen_train,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=data_gen_valid,
                        callbacks=[tensorboard, checkpoint])
    newhist = pd.DataFrame(history.history)
    newhist["epoch"] = history.epoch

    #  --- Output ---
    model.save("{}.h5".format(os.path.join(LOGDIR, NEWNAME)))
    newhist.to_csv("{}.cvs".format(os.path.join(LOGDIR, NEWNAME)))
