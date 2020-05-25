# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : breaking_classifier.py
# POURPOSE : classify wave breaking using a relatively simple convnet
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : XX/XX/XXXX [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import os

import argparse
import numpy as np

import itertools

from glob import glob

import datetime

from skimage.io import imread

import pandas as pd

from sklearn.model_selection import train_test_split

import io

import platform

from tensorflow import summary
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv2D, Dropout,
                                     MaxPooling2D, Flatten, Input)
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

from tensorflow import image, expand_dims

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "#E9E9F1",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
                   decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def log_confusion_matrix(epoch, logs):
    """Compute the confusion matrix."""

    y_pred = np.squeeze(model.predict_classes(X_valid))

    # Calculate the confusion matrix.
    cm = confusion_matrix(y_pred, y_valid)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=["Passive", "Active"])
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        summary.image("Confusion Matrix", cm_image, step=epoch)


def plot_to_image(figure):
    """Convert the matplotlib plot specified by 'figure' to a PNG image and
       returns it. The supplied figure is then closed and inaccessible."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    img = image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    img = expand_dims(img, 0)
    return img


def vgg_block(layer_in, n_filters, n_conv, dropout=0.25):
    """Define a VGG block with dropout."""
    # add convolutional layers
    for _ in range(n_conv):
        layer_in = Conv2D(n_filters, (3, 3), padding='same',
                          activation='relu')(layer_in)
    # add max pooling layer
    layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
    layer_in = Dropout(dropout)(layer_in)

    return layer_in


if __name__ == "__main__":

    print("\nClassifiying wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--labels", "-l",
                        nargs=1,
                        action="store",
                        dest="labels",
                        required=True,
                        help="Input csv file with labels."
                             "One of the colmns must be named 'class'.",)

    # input configuration file
    parser.add_argument("--data", "-data",
                        nargs=1,
                        action="store",
                        dest="data",
                        required=True,
                        help="Input path with image data.",)

    # input configuration file
    parser.add_argument("--log-dir", "--logdir",
                        nargs=1,
                        action="store",
                        dest="logdir",
                        required=True,
                        help="Path where to store the logs.",)

    # random state
    parser.add_argument("--random-state",
                        nargs=1,
                        action="store",
                        dest="random_state",
                        required=False,
                        default=[11],
                        help="Random state for reproducibility.",)

    parser.add_argument("--test-size",
                        nargs=1,
                        action="store",
                        dest="test_size",
                        required=False,
                        default=[0.25],
                        help="Fraction of the dataset to use as test.",)

    parser.add_argument("--learning-rate", "-lr",
                        nargs=1,
                        action="store",
                        dest="learning_rate",
                        required=False,
                        default=[0.0001],
                        help="Learning rate.",)

    parser.add_argument("--epochs",
                        nargs=1,
                        action="store",
                        dest="epochs",
                        required=False,
                        default=[200],
                        help="Number of epochs.",)

    parser.add_argument("--batch-size",
                        nargs=1,
                        action="store",
                        dest="batch_size",
                        required=False,
                        default=[128],
                        help="Batch size.",)

    parser.add_argument("--dropout",
                        nargs=1,
                        action="store",
                        dest="dropout",
                        required=False,
                        default=[0.25],
                        help="Dropout rate.",)

    parser.add_argument("--target-labels",
                        nargs="*",
                        action="store",
                        dest="target_labels",
                        default=[4, 5],
                        required=False,
                        help="Target labels to consider as wave breaking.",)

    parser.add_argument("--basename", "-b",
                        nargs=1,
                        action="store",
                        dest="basename",
                        default=["cnn"],
                        required=False,
                        help="Model basename.",)

    args = parser.parse_args()

    # Constants
    RANDOM_STATE = int(args.random_state[0])
    TEST_SIZE = float(args.test_size[0])
    TRAIN_SIZE = 1 - TEST_SIZE
    LEARNING_RATE = float(args.learning_rate[0])
    EPOCHS = int(args.epochs[0])
    BATCH_SIZE = int(args.batch_size[0])
    DROPOUT = float(args.dropout[0])

    METRICS = [metrics.TruePositives(name="True_Positives"),
               metrics.FalsePositives(name="False_Positives"),
               metrics.TrueNegatives(name="True_Negatives"),
               metrics.FalseNegatives(name="False_Negatives"),
               metrics.BinaryAccuracy(name="Binary_Accuracy"),
               metrics.Precision(name="Precision"),
               metrics.Recall(name="Recall"),
               metrics.AUC(name="AUC")]

    # main()

    print("\n - Pre-processing data")

    # --- callbacks ---

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if platform.system().lower() == "windows":
        logdir = args.logdir[0] + "\\" + date
    else:
        logdir = args.logdir[0] + "/" + date
    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)

    # log confusion matrix
    if platform.system().lower() == "windows":
        file_writer_cm = summary.create_file_writer(logdir + "\\" + "CM")
    else:
        file_writer_cm = summary.create_file_writer(logdir + "/" + "CM")

    tensorboard = callbacks.TensorBoard(log_dir=logdir,
                                        histogram_freq=1,
                                        profile_batch=1)

    CM = callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    # --- data input ---

    # load the labels
    input = args.labels[0]
    df = pd.read_csv(input)
    try:
        labels = df["class"].values
    except Exception:
        raise ValueError("column \"class\" not present in the dataframe.")

    # binarize labels
    if len(np.unique(labels)) > 2:
        print("    warning: more than 2 labels in the dataset.")

        for l in np.array(args.target_labels):
            idxs = np.where(labels == l)[0]
            labels[idxs] = 1
        labels[labels != 1] = 0

    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.data[0]
    if os.path.isdir(inp):
        frame_path = os.path.abspath(inp)
        if platform.system().lower() == "windows":
            frames = glob(inp + "\\*")
        else:
            frames = glob(inp + "/*")
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))



    # load and group the image data
    X = []
    for fname in frames:
        X.append(imread(fname) / 255)
    X = np.array(X)

    # reshape to a shape that tensorflow likes
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, random_state=RANDOM_STATE, stratify=labels,
        test_size=TEST_SIZE, train_size=TRAIN_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, random_state=RANDOM_STATE,
        test_size=TEST_SIZE, train_size=TRAIN_SIZE)

    # --- write data to tensorboard ---

    file_writer = summary.create_file_writer(logdir)

    # log image data
    with file_writer.as_default():
        summary.image("Train data",
                      X_train, max_outputs=X_train.shape[0], step=0)
        summary.image("Test data",
                      X_test, max_outputs=X_test.shape[0], step=0)

    # --- data augumentation ---

    # flip, shift, rotate and zoom
    datagen = ImageDataGenerator(zoom_range=0.2,
                                 rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)
    datagen.fit(X_train)

    # --- model definition ---

    print("\n - Training")

    # define model input
    visible = Input(shape=(X.shape[1], X.shape[2], 1))
    # add vgg module
    layer = vgg_block(visible, 64, 2, DROPOUT)
    # add vgg module
    layer = vgg_block(layer, 128, 4, DROPOUT)
    # add vgg module
    layer = vgg_block(layer, 256, 8, DROPOUT)

    # flatten
    flat = Flatten()(layer)
    fcn1 = Dense(2048)(flat)
    drop1 = Dropout(DROPOUT)(fcn1)
    fcn2 = Dense(1024)(drop1)
    drop2 = Dropout(DROPOUT)(fcn2)
    fcn3 = Dense(1)(drop2)

    # create model
    model = Model(inputs=visible, outputs=fcn3)

    # compile the model
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",  # try categorical_hinge later
                  metrics=METRICS)

    model.summary()

    # --- train ---

    history = model.fit_generator(datagen.flow(X_train,
                                               y_train,
                                               batch_size=BATCH_SIZE),
                                  steps_per_epoch=len(X_train) // BATCH_SIZE,
                                  validation_steps=len(X_valid) // BATCH_SIZE,
                                  epochs=EPOCHS,
                                  verbose=1,
                                  validation_data=(X_valid, y_valid),
                                  callbacks=[tensorboard, CM])
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    # save model to file
    print("\n - Saving model to file")
    model.save("{}.h5".format(os.path.join(logdir, args.basename[0])))

    # predict
    y_pred = np.squeeze(model.predict_classes(X_test))

    print("\n")
    print(classification_report(y_test, y_pred))

    print("\nMy work is done!\n")
