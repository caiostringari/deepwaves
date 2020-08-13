"""

Test the classifier.

The data needs to be in a folder called "test" which has
sub-folders "0" and "1"

For example:

test
    ├───0
    ├───1

You will need to download a pre-trained model

# PROGRAM   : test.py
# POURPOSE  : classify wave breaking using a convnets
# AUTHOR    : Caio Eadi Stringari
# EMAIL     : caio.stringari@gmail.com
# V1.0      : 05/05/2020 [Caio Stringari]

"""
import argparse

import numpy as np

import tensorflow as tf

import pandas as pd

import pathlib

from sklearn.metrics import classification_report

from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':

    print("\nClassifiying wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--model", "-M",
                        nargs=1,
                        action="store",
                        dest="model",
                        required=True,
                        help="Input model in .h5 format.",)

    # input model
    parser.add_argument("--data", "-data",
                        nargs=1,
                        action="store",
                        dest="data",
                        required=True,
                        help="Input path with image data.",)

    parser.add_argument("--threshold", "-trx",
                        nargs=1,
                        action="store",
                        dest="TRX",
                        default=[0.5],
                        required=False,
                        help="Probability threshold for classification.")

    # output model
    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output file.",)

    args = parser.parse_args()

    # --- test data input ---
    test_dir = args.data[0]
    test_dir = pathlib.Path(test_dir)
    image_count = len(list(test_dir.glob('*/*')))

    BATCH_SIZE = int(image_count/10)

    class_names = np.array([item.name for item in test_dir.glob('*')])

    try:
        nclasses = len(class_names)
        print("  Found image data, proceeding.\n")
        print("   - Classes are {}".format(class_names))
    except Exception:
        raise IOError("Check your data!")

    # --- model ---
    model = tf.keras.models.load_model(args.model[0])

    inp_shape = model.input_shape
    img_height = inp_shape[1]  # image height for all images
    img_width = inp_shape[2]  # image width for all images

    datagen = ImageDataGenerator(rescale=1./255.)

    print("\n    Fitting the teset data generator:\n")
    data_gen_test = datagen.flow_from_directory(
        directory=str(test_dir), batch_size=BATCH_SIZE, shuffle=False,
        target_size=(img_height, img_width), classes=["0", "1"],
        class_mode="binary")

    # predict on the test data
    print("\n    Prediction loop:\n")
    ytrue = []
    yhat = []
    for step in range(data_gen_test.n // BATCH_SIZE):
        print("     - step {} of {}".format(
            step+1, data_gen_test.n // BATCH_SIZE), end="\r")
        X, y = data_gen_test.next()
        yh = model.predict(X)
        for i, j in zip(np.squeeze(y), np.squeeze(yh)):
            ytrue.append(i)
            yhat.append(j)

    # predicted labels
    TRX = float(args.TRX[0])
    yhat = np.squeeze(yhat)
    ypred = np.zeros(yhat.shape)
    ypred[yhat > TRX] = 1

    df = pd.DataFrame(np.vstack([ytrue, ypred]).T,
                      columns=["true", "prediction"])
    df.to_csv(args.output[0], index=False)

    print("\nMy work is done!\n")
