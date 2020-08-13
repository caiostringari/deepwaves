"""
Sumarize results for the train/valid/test splits.

# PROGRAM   : metrics.py
# POURPOSE  : compute model metrics on the test datasete
# AUTHOR    : Caio Eadi Stringari
# EMAIL     : caio.stringari@gmail.com
# V1.0      : 05/05/2020 [Caio Stringari]
"""
import argparse

import numpy as np

import tensorflow as tf

import pandas as pd

import pathlib

try:
    import efficientnet.tfkeras as efn
except Exception:
    print(ImportError("\nWarning: run pip install -U --pre efficientnet"))


from tensorflow.keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':

    print("\nClassifiying wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input model and history
    parser.add_argument("--model", "-M",
                        nargs=1,
                        action="store",
                        dest="model",
                        required=True,
                        help="Input model in .h5 format.",)

    parser.add_argument("--history", "-hist",
                        nargs=1,
                        action="store",
                        dest="history",
                        required=True,
                        help="Input model history in csv format.",)

    # input test data
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

    parser.add_argument("--epoch", "-epch",
                        nargs=1,
                        action="store",
                        dest="epoch",
                        default=[-1],
                        required=False,
                        help="Which epoch to use. Default is last epoch.")

    # output data
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
    epoch = int(args.epoch[0])

    BATCH_SIZE = int(image_count/10)

    class_names = np.array([item.name for item in test_dir.glob('*')])

    try:
        nclasses = len(class_names)
        print("  Found image data, proceeding.\n")
        print("   - Classes are {}".format(class_names))
    except Exception:
        raise IOError("Check your data!")

    # --- pre-trained model ---
    model = tf.keras.models.load_model(args.model[0])
    history = pd.read_csv(args.history[0])

    # train data
    accuracy = history.iloc[epoch]["Binary_Accuracy"]
    tp = history.iloc[epoch]["True_Positives"]
    fp = history.iloc[epoch]["False_Positives"]
    tn = history.iloc[epoch]["True_Negatives"]
    fn = history.iloc[epoch]["False_Negatives"]
    precision = history.iloc[epoch]["Precision"]
    recall = history.iloc[epoch]["Recall"]
    auc = history.iloc[epoch]["AUC"]

    X = [accuracy, tp, fp, tn, fn, precision, recall, auc]
    cols = ["Binary_Accuracy", "True_Positives", "False_Positives",
            "True_Negatives",  "False_Negatives", "Precision", "Recall", "AUC"]
    df_train = pd.DataFrame([X], columns=cols)
    df_train.index = ["Train"]
    print(df_train)

    # validation data
    accuracy = history.iloc[epoch]["val_Binary_Accuracy"]
    tp = history.iloc[epoch]["val_True_Positives"]
    fp = history.iloc[epoch]["val_False_Positives"]
    tn = history.iloc[epoch]["val_True_Negatives"]
    fn = history.iloc[epoch]["val_False_Negatives"]
    precision = history.iloc[epoch]["val_Precision"]
    recall = history.iloc[epoch]["val_Recall"]
    auc = history.iloc[epoch]["val_AUC"]

    X = [accuracy, tp, fp, tn, fn, precision, recall, auc]
    cols = ["Binary_Accuracy", "True_Positives", "False_Positives",
            "True_Negatives",  "False_Negatives", "Precision", "Recall", "AUC"]
    df_val = pd.DataFrame([X], columns=cols)
    df_val.index = ["Validation"]
    print(df_val)

    # evaluate the model on test data
    inp_shape = model.input_shape
    img_height = inp_shape[1]  # image height for all images
    img_width = inp_shape[2]  # image width for all images

    datagen = ImageDataGenerator(rescale=1./255.)

    print("\n    Fitting the teset data generator:\n")
    data_gen_test = datagen.flow_from_directory(
        directory=str(test_dir), batch_size=BATCH_SIZE, shuffle=False,
        target_size=(img_height, img_width), classes=["0", "1"],
        class_mode="binary")

    result = model.evaluate(data_gen_test)
    metrics = dict(zip(model.metrics_names, result))

    # validation data
    accuracy = metrics["Binary_Accuracy"]
    tp = metrics["True_Positives"]
    fp = metrics["False_Positives"]
    tn = metrics["True_Negatives"]
    fn = metrics["False_Negatives"]
    precision = metrics["Precision"]
    recall = metrics["Recall"]
    auc = metrics["AUC"]

    X = [accuracy, tp, fp, tn, fn, precision, recall, auc]
    cols = ["Binary_Accuracy", "True_Positives", "False_Positives",
            "True_Negatives",  "False_Negatives", "Precision", "Recall", "AUC"]
    df_test = pd.DataFrame([X], columns=cols)
    df_test.index = ["Test"]

    # merge results
    df = pd.concat([df_train, df_val, df_test])

    print(df)

    df.to_excel(args.output[0], float_format="%.3f", index=True)

    print("\nMy work is done!\n")
