"""
PROGRAM   : predict.py
POURPOSE  : classify wave breaking using a convnet
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V1.0      : 16/07/2020 [Caio Stringari]

Predict Active Wave Breaking

All you need is a pre-trained model and a series of images

For example:

folder
    ├───images
        ├───img_0001.png
        ├───img_0002.png
        ├───...
        ├───img_000X.png


You will need to download a pre-trained model if you don't have one.

Trained on 10k samples:
https://drive.google.com/file/d/1FOXj-uJdXtyzxOVRHHuj1X8Xx0B8jaiT/view?usp=sharing
"""
import argparse

import numpy as np

import tensorflow as tf

import pandas as pd

import pathlib

from os.path import basename

from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':

    print("\nClassifiying wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--model", "-M",
                        action="store",
                        dest="model",
                        required=True,
                        help="Input model in .h5 format.",)

    # input model
    parser.add_argument("--data", "-data",
                        action="store",
                        dest="data",
                        required=True,
                        help="Input path with image data.",)

    parser.add_argument("--threshold", "-trx",
                        action="store",
                        dest="TRX",
                        default=0.5,
                        required=False,
                        help="Probability threshold for classification.")

    # output model
    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output file (csv).",)

    args = parser.parse_args()

    # --- data input ---
    data_dir = args.data
    data_dir = pathlib.Path(data_dir)

    # Fix batch_size at 1. Waste of resources but makes my life easier
    BATCH_SIZE = 1

    # --- model ---
    model = tf.keras.models.load_model(args.model)

    inp_shape = model.input_shape
    img_height = inp_shape[1]  # image height for all images
    img_width = inp_shape[2]  # image width for all images

    datagen = ImageDataGenerator(rescale=1. / 255.)

    print("\n    Fitting the teset data generator:\n")
    data_gen = datagen.flow_from_directory(
        directory=str(data_dir), batch_size=BATCH_SIZE, shuffle=False,
        target_size=(img_height, img_width), class_mode='binary')

    # predict on the test data
    print("\n    Prediction loop:\n")
    probs = []
    files = []
    k = 0
    for step in range(data_gen.n // BATCH_SIZE):
        print("     - step {} of {}".format(
            step + 1, data_gen.n // BATCH_SIZE), end="\r")

        # classify
        X, y = data_gen.next()
        yh = model.predict(X)
        probs.append(yh)

        # file name
        fname = basename(data_gen.filenames[k])
        files.append(fname)

        k += 1

    # predicted labels
    TRX = float(args.TRX)
    yhat = np.squeeze(probs)
    ypred = np.zeros(yhat.shape)
    ypred[yhat > TRX] = 1

    # build a dataframe
    df = pd.DataFrame(ypred, columns=["label"])
    df["file"] = files

    # save
    if args.output.endswith("csv"):
        df.to_csv(args.output)
    else:
        df.to_csv(args.output + ".csv")
