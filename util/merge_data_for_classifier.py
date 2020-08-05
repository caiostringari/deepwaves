# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : merge_data_for_classifier.py
# POURPOSE : TODO: Update
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : XX/XX/XXXX [Caio Stringari]
#
# TODO: VERFIRY THE OPUTS OF THIS SCRIPT!
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import os
import argparse

import random
import string

from glob import glob
from natsort import natsorted

import numpy as np
from skimage.io import imread, imsave
from skimage.color import grey2rgb

import pandas as pd


def random_string(length=16):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


if __name__ == "__main__":

    print("\nExtracting data for the classifier, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i",
                        nargs="*",
                        action="store",
                        dest="input",
                        required=True,
                        help="Input processed folders.",)

    parser.add_argument("--crop-size",
                        nargs=1,
                        action="store",
                        dest="crop_size",
                        default=[None],
                        required=False,
                        help="Output size.",)

    parser.add_argument("--target-labels",
                        nargs="*",
                        action="store",
                        dest="target_labels",
                        default=[4, 5],
                        required=False,
                        help="Target labels to consider as wave breaking.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["classifier/"],
                        required=False,
                        help="Output path.",)

    args = parser.parse_args()

    # input paths
    paths = args.input

    # output path
    out = args.output[0]
    os.makedirs(out, exist_ok=True)

    print("\nProcessing labels:\n")
    dfs = []
    for i, path in enumerate(paths):
        if os.path.isdir(path):
            print("Processing path {}".format(path))

            # read csv file
            xls = glob(path+"/*.xlsx")
            if xls:
                print(" + labels found")
                df = pd.read_excel(xls[0])
                dfs.append(df)
    df = pd.concat(dfs)

    # binarize labels
    try:
        labels = df["label"].values
        labels_str = np.ones(labels.shape).astype(str)
    except Exception:
        raise ValueError("At least one column must be called \'{label}\'.")
    if len(np.unique(labels)) > 2:
        print("- Warning: more than 2 unique labels in the dataset.")
        print("  using \"TARGET_LABELS\" to binarize the dataset.")

        for l in np.array(args.target_labels).astype(np.int):
            idxs = np.where(labels == l)[0]
            labels_str[idxs] = "breaking"
        idxs = np.where(labels_str != "breaking")[0]
        labels_str[idxs] = "otherwise"

    # creat a folder for each labels
    folder0 = os.path.join(args.output[0], "0")
    folder1 = os.path.join(args.output[0], "1")
    os.makedirs(folder0, exist_ok=True)
    os.makedirs(folder1, exist_ok=True)

    # loop over images
    print("\n\nProcessing images:")
    fnames = []
    for i, path in enumerate(paths):
        if os.path.isdir(path):
            print("Processing path {}".format(path))
            pngs = natsorted(glob(path+"/img/*.png"))
            if pngs:
                print(" + images found")
                for png in pngs:
                    fnames.append(png)
            else:
                raise IOError("  - Did not find any images!")

    # save data to a temporary folder so that we can use keras data generators
    # make sure that the data has 3 chanels
    k = 0
    i = 0
    j = 0
    print("\n")
    for label, fname in zip(labels_str, fnames):
        print("   - Processing image {} of {}".format(k+1, len(fnames)),
              end="\r")
        if label == "otherwise":
            img3c = grey2rgb(imread(fname))  # make shure that there are 3d
            fname0 = random_string()+".png"
            imsave(os.path.join(folder0, fname0), img3c)
            i += 1
        elif label == "breaking":
            img3c = grey2rgb(imread(fname))  # make shure that there are 3d
            fname1 = random_string()+".png"
            imsave(os.path.join(folder1, fname1), img3c)
            j += 1
        else:
            raise ValueError("Fatal, stopping now.")
        k += 1

    print("\n\nMy work is done!\n")
