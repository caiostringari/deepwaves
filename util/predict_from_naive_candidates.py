# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : predict_from_naive_candidates.py
# POURPOSE : predict if a wave is actively breaking with a pre-trained
#            classifier and results from the naive search.
# AUTHOR   : Caio Eadi Stringari
# V1.0     : 15/04/2020
# V2.0     : 05/08/2020
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
r"""
Detect wave active breaking using a machine learning approach.

NOTE: Get a pre-trained model before trying to use this script!

Usage:
-----
python predict_from_naive_candidates.py --help

Example:
-------
python predict_from_naive_candidates.py --debug \
                                          --model "path/to/model.h5 \
                                          --threshold 0.5
                                          --input path/to/frames/ \
                                          --roi path/to/roi.csv \
                                          --output output.csv \
                                          --frames-to-plot 200 \
                                          --size 128 128

Explanation:
-----------

--debug : runs in debug mode, will save output plots

--model : pre-trained tensorflow model.

--threshold : threshold used for the activation of the sigmoid function.

-i : input path with images

-o : output file name (see below for explanation)

--temporary-path : path to write temporary files and/or plots if in debug mode

--from-frame : at which sequential frame to start

--number-of-frames : number of frames to use

--regex : regex used to get frame file names

Output:
------

The output CSV columns are organized are the same as described in
"naive_wave_breaking_detector"

The only addition is:
    - class : Event classification (0 for unbroken, 1 for breaking)
"""

import os
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import warnings

import argparse
import numpy as np

import re

from glob import glob
from natsort import natsorted

from skimage.io import imread

import tensorflow as tf

import pandas as pd

from copy import copy

from skimage.color import grey2rgb
from skimage.transform import resize

# used only for debug
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# set a pandas option
pd.set_option("mode.chained_assignment", None)

# catch an annoying future warning from numpy that will never be fixed
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def compute_roi(roi, frame_path, regex="[0-9]{6,}"):
    """
    Compute  the region of interest (ROI) and a mask.

    Input can be either a list of coordinates or a dataframe.

    Note the the format expected by this funcion is the same as what
    matplotlib.patches.Rectangle expects and is in image coordinates.

    roi = [top_left_corner, top_right_corner, length, height]

    Parameters:
    ----------
    roi : list, pandas.DataFrame, bool
        Either a list or a pandas dataframe.
    frame_path : str
        A valid path pointing to a image file
    regex : str
        Regex to get sequential frame numbers.

    Returns:
    -------
    roi_coords : list
        A list of coordinates
    rec_patch :  matplotlib.patches.Rectangle
        A Rectangle instance of the ROI
    mask : np.array
        A image array with everything outside the ROI masked
    """

    # if it is a dataframe
    if isinstance(roi, pd.DataFrame):

        # select frame
        # idx = int(os.path.basename(frame_path).split(".")[0])

        # try to figure out frame number
        res = re.search(regex, os.path.basename(frame_path))
        idx = int(res.group())

        roi = roi.loc[roi["frame"] == idx]
        roi = [int(roi["i"]), int(roi["j"]),
               int(roi["width"]), int(roi["height"])]

        # update mask and rectangle
        img = imread(frame_path)
        mask = np.zeros(img.shape)
        mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1
        rec_patch = patches.Rectangle((int(roi[0]), int(roi[1])),
                                      int(roi[2]), int(roi[3]),
                                      linewidth=2,
                                      edgecolor="deepskyblue",
                                      facecolor="none",
                                      linestyle="--")
    # if it is not a dataframe
    else:
        img = imread(frame_path)

        # if it is a list
        if isinstance(roi, list):
            mask = np.zeros(img.shape)
            mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1
            rec_patch = patches.Rectangle((int(roi[0]), int(roi[1])),
                                          int(roi[2]), int(roi[3]),
                                          linewidth=2,
                                          edgecolor="r",
                                          facecolor="none",
                                          linestyle="--")
        # cant deal with it, asign all False
        else:
            roi = [False, False, False, False]
            img = imread(frame_path)
            mask = np.ones(img.shape)
            rec_patch = False

    # coordinates
    roi_coords = [roi[0], (roi[0] + roi[2]), roi[1],  (roi[1] + roi[3])]

    return roi_coords, rec_patch, mask


def plot(frmid, frm, df, roi=False, total_frames=-1, temp_path="tmp"):
    """
    Plot the results of the classification.

    Parameters:
    ----------
    frmid : int
        frame id. Usuallya a sequential number
    frm : np.ndarray
        a valid array for plt.imshow()
    df : pd.DataFrame
        dataframe with the classification results
    roi : mpl.patches.Rectangle
        region of interest poligon.
    total_frames : int
        total number of frames
    temp_path : path-like
        path to where to save the plots


    Returns:
    -------
    None. Will write to disk.
    """

    # open a new figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   sharex=True,
                                   sharey=True)

    ax1.imshow(frm, cmap="Greys_r")
    ax2.imshow(frm, cmap="Greys_r")

    ntotal = 0
    for r, row in df.iterrows():
        # if radii are the same, draw a circle
        try:
            r1 = row["ir"]
            r2 = row["jr"]
        except Exception:
            r1 = row["ir"]
            r2 = r1
        if r1 == r2:
            c = patches.Circle((row["ic"], row["jc"]), r1,
                               edgecolor="lawngreen", facecolor="None")
        # if not, draw a ellipse
        else:
            c = patches.Ellipse((row["ic"], row["jc"]),
                                row["ir"]*2, row["jr"]*2,
                                angle=row["theta_ij"],
                                facecolor="None",
                                edgecolor="lawngreen",
                                linewidth=1)
        ax1.add_patch(copy(c))
        ntotal += 1

    # plot active breaking only
    nrboust = 0
    for r, row in df.iterrows():
        if row["class"] == 1:
            # if radii are the same, draw a circle
            try:
                r1 = row["ir"]
                r2 = row["jr"]
            except Exception:
                r1 = row["ir"]
                r2 = r1
            if r1 == r2:
                c = patches.Circle((row["ic"], row["jc"]), r1,
                                   edgecolor="lawngreen",
                                   facecolor="None")
            # if not, draw a ellipse
            else:
                c = patches.Ellipse((row["ic"], row["jc"]),
                                    row["ir"]*2, row["jr"]*2,
                                    angle=row["theta_ij"],
                                    facecolor="None",
                                    edgecolor="lawngreen",
                                    linewidth=1)
            ax2.add_patch(copy(c))
            nrboust += 1

    # annotations
    for ax, n in zip([ax1, ax2], [ntotal, nrboust]):

        txt = "Frame {} of {}".format(frmid, total_frames)
        ax.text(0.05, 0.95, txt, color="deepskyblue",
                va="top", zorder=100, transform=ax.transAxes,
                ha="left", fontsize=12,
                bbox=dict(boxstyle="square", ec="none", fc="0.1",
                          lw=1, alpha=0.7))

        txt = "Number of clusters: {} ".format(n)
        ax.text(1 - 0.05, 1 - 0.95, txt, color="deepskyblue",
                va="bottom", zorder=100, transform=ax.transAxes,
                ha="right", fontsize=12,
                bbox=dict(boxstyle="square", ec="none", fc="0.1",
                          lw=1, alpha=0.7))

        if roi:
            ax.add_patch(copy(roi))

        ax.set_xlabel("$i$ $[pixel]$")

    ax1.set_ylabel("$j$ $[pixel]$")

    ax1.set_title("Naive Detector")
    ax2.set_title("Naive Detector + Neural Net")

    fig.tight_layout()
    figname = str(frmid).zfill(8) + "." + args.image_format[0]
    plt.savefig(os.path.join(temp_path, figname),
                bbox_inches="tight", pad_inches=0.1,
                dpi=200)
    plt.close()

    return None


def main():
    """Call the main program."""
    TRX = float(args.trx[0])  # treshold for binary classifier

    # create the output path, if not present
    temp_path = os.path.abspath(args.temp_path[0])
    os.makedirs(temp_path, exist_ok=True)

    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.frames[0]
    if os.path.isdir(inp):
        frames = natsorted(glob(inp + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # load the model
    print("\n  Loading tensorflow model, please wait...")
    model = tf.keras.models.load_model(args.model[0])

    # get the input shape for cropping the candidates
    inp_shape = model.input_shape
    print("  Model loaded.")

    # load wave breaking candidates
    df = pd.read_csv(args.input[0])
    # drop any nans
    df = df.dropna()

    # ckeck if all needed keys are present
    targets = ["ic", "jc", "ir", "frame", "theta_ij"]
    for t in targets:
        if t not in df.keys():
            raise ValueError(
                "Key \"{}\" must be present in the data.".format(t))

    # input image size
    ISIZE = (int(args.size[0]), int(args.size[1]))

    # handle region of interest
    if args.roi[0]:
        is_roi_file = os.path.isfile(args.roi[0])
    if args.roi[0]:
        if is_roi_file:
            roi = pd.read_csv(args.roi[0])
            # fill nans with the previous/next valid values
            roi = roi.fillna(method="bfill")
            roi = roi.fillna(method="ffill")

            # check sizes
            if len(roi) != len(frames):
                mframes = min(len(roi), len(frames))
                print("  \nwarning: number of frames does not match number of"
                      " of rows in the ROI file. Setting number of frames"
                      " to: {}".format(mframes))

                # cut the lists to size
                frames = frames[0:mframes]
                roi = roi.iloc[0:mframes]
            else:
                pass
        else:
            roi = False
            raise ValueError("Could not process region-of-interest file.")
    else:
        roi = False

    # slice frames to requested range
    start = int(args.start_frame[0])
    if int(args.nframes[0]) > 0:
        end = start + int(args.nframes[0])
    else:
        end = len(frames)
    frames = frames[start:end]

    start_df = df["frame"].min()
    if start_df < start:
        print("Warning: First frame in the dataframe is {}, user asked to start from {}.".format(start_df, start))

    # loop over frames
    print("\n  Looping over frames")
    dfs = []
    for k, fname in enumerate(frames):
        print("   -  processing frame {} of {}".format(k + 1, len(frames)))

        # load the frame
        frm = plt.imread(fname)
        try:
            frm = frm[:, :, 0]
        except Exception:
            frm = frm[:, :]

        # frame size
        try:
            height, width, c = frm.shape
        except Exception:
            height, width = frm.shape

        # locate the current frame in the breaking candidates dataframe
        res = re.search(args.regex[0], os.path.basename(fname))
        frm_id = int(res.group())

        df_frm = df.loc[df["frame"] == frm_id]

        # if there is a detection for that frame, apply the pipeline
        indexes = []  # valid dataframe indexes
        if not df_frm.empty:

            # crop to sub-sets and prepare for classification
            X = []  # classifier tensor
            for i, row in df_frm.iterrows():

                # crop iamge
                left = int(row["ic"] - (ISIZE[0] / 2))
                right = int(row["ic"] + (ISIZE[0] / 2))
                top = int(row["jc"] + (ISIZE[1] / 2))
                bottom = int(row["jc"] - (ISIZE[1] / 2))

                # check if the croped image is compatible with the whole frame
                if left < 0:
                    left = 0
                if right > width:
                    right = width
                if top < 0:
                    top = 0
                if top > height:
                    top = height
                if bottom < 0:
                    bottom = 0

                # crop and go to 3 channels, if needed
                crop = grey2rgb(frm[bottom:top, left:right])

                # check if the croped section is usable. if any of the dims
                # is zero, it is not.
                usable = True
                if crop.shape[0] == 0:
                    if crop.shape[1] == 0:
                        usable = False

                if usable:
                    # resize to match what the TF model wants
                    reshape = False
                    if crop.shape[0] != inp_shape[0]:
                        reshape = True
                    if crop.shape[1] != inp_shape[1]:
                        reshape = True
                    if reshape:
                        try:
                            crop = resize(crop, (inp_shape[1], inp_shape[2]))
                            X.append(crop)
                            indexes.append(i)
                        except Exception:
                            # could not use this data for the love of god.
                            pass

            # cast croped images and dataframe to proper shapes
            X = np.array(X)
            df_frm = df_frm.iloc[np.where(df_frm.index.values == indexes)]

            # apply the model
            yhat = np.squeeze(model.predict(X))

            # 0 means unbroken
            # 1 means broken

            classes = np.zeros(X.shape[0])  # starts as unbroken

            # if p(X(i)) greater than TRX (0.5 per default), assigns 1
            classes[yhat > TRX] = 1

            # update dataframe
            df_frm["class"] = classes

            # append to output
            dfs.append(df_frm)

        # plot
        if args.debug:

            # try to compute the ROI
            try:
                _, roi_rect, _ = compute_roi(roi, fname)
            except Exception:
                raise
                roi_rect = False

            try:
                # print(frm.shape, frm.min(), frm.max())
                plot(frm_id, frm, df_frm, roi=roi_rect,
                     total_frames=len(frames), temp_path=args.temp_path[0])
            except Exception:
                print("Warning: Could not plot frame {} of {}".format(
                    k, len(frames)))

    # output
    dfout = pd.concat(dfs)
    dfout.to_csv(args.output[0], index=False)


if __name__ == "__main__":

    print("\nDetecting active wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input detected wave breaking candidates.",)

    parser.add_argument("--regex", "-re", "-regex",
                        nargs=1,
                        action="store",
                        dest="regex",
                        required=False,
                        default=["[0-9]{6,}"],
                        help="Regex to search for frames. Default is [0-9]{6,}.",)

    parser.add_argument("--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="frames",
                        required=True,
                        help="Input path with image data.",)

    parser.add_argument("--region-of-interest", "-roi", "--roi",
                        nargs="*",
                        action="store",
                        dest="roi",
                        default=[False],
                        required=False,
                        help="Region of interest. Must be a file generated "
                             "with minmun_bounding_geometry.py",)

    parser.add_argument("--size", "-size",
                        nargs=2,
                        action="store",
                        dest="size",
                        default=[256, 256],
                        required=False,
                        help="Image size.",)

    parser.add_argument("--model", "-m",
                        nargs=1,
                        action="store",
                        dest="model",
                        required=True,
                        help="Your pre-trained model.",)

    parser.add_argument("--threshold", "-trx",
                        nargs=1,
                        action="store",
                        dest="trx",
                        default=[0.5],
                        required=False,
                        help="Treshold for the binary classifier.",)

    parser.add_argument("--debug", "-debug",
                        action="store_true",
                        dest="debug",
                        help="Debug plots.",)

    parser.add_argument("--frames-to-process", "-nframes", "--nframes",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        default=[-1],
                        help="How many frames to plot.",)

    parser.add_argument("--from-frame", "-start", "-from-frame",
                        nargs=1,
                        action="store",
                        dest="start_frame",
                        default=[0],
                        help="At which frame to start.",)

    parser.add_argument("--image-format",
                        nargs=1,
                        action="store",
                        dest="image_format",
                        default=["jpg"],
                        help="In what format to save the output image.",)

    parser.add_argument("--temporary-path", "-temporary-path",
                        nargs=1,
                        action="store",
                        dest="temp_path",
                        default=["dbg/"],
                        required=False,
                        help="Debug images folder name.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["active_wave_breaking.csv"],
                        required=False,
                        help="Output file.",)

    args = parser.parse_args()

    main()

    print("\nMy work is done!\n")
