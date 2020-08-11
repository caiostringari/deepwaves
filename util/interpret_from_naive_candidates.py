# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : interpret_from_naive_candidates.py
# POURPOSE : interpret using GradCAM if a wave is actively breaking with a pre-trained
#            classifier and results from the naive search.
# AUTHOR   : Caio Eadi Stringari
# V1.0     : 15/04/2020
# V2.0     : 05/08/2020
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
r"""

Interpret using GradCAM if a wave is actively breaking with a pre-trained
classifier and results from the naive search.

NOTE: Get a pre-trained model before trying to use this script!

Usage:
-----
python interpret_from_naive_candidates.py --help

Example:
-------
python interpret_from_naive_candidates.py --model "path/to/model.h5 \
                                          --threshold 0.5
                                          --input path/to/frames/ \
                                          --roi path/to/roi.csv \
                                          --output output.csv \
                                          --frames-to-plot 200 \
                                          --size 128 128

Explanation:
-----------

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

import tensorflow as tf
from tensorflow import keras

import argparse
import numpy as np

import re

from glob import glob
from natsort import natsorted

import cv2
from skimage.io import imread

import pandas as pd

from copy import copy

from skimage.color import grey2rgb
from skimage.transform import resize

# used only for debug
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


class GradCAM:
    """
    Implements GradCAM.

    reference: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    """

    def __init__(self, model, layerName):
        """Initialize the model."""
        self.model = model
        self.layerName = layerName

        self.gradModel = keras.models.Model(inputs=[self.model.inputs],
                                            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

    def compute_heatmap(self, image, classIdx, eps=1e-8):
        """Compute a heatmap with the class activation."""
        with tf.GradientTape() as tape:
            tape.watch(self.gradModel.get_layer(self.layerName).output)
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(inputs)

            if len(predictions) == 1:
                # binary Classification
                loss = predictions[0]
            else:
                loss = predictions[:, classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap


if __name__ == "__main__":

    print("\n Creating GradCAM animation, please wait...\n")

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

    parser.add_argument("--vmin", "-vmin",
                        nargs=1,
                        action="store",
                        dest="vmin",
                        default=[127],
                        required=False,
                        help="Minimum value for the colormap.",)

    parser.add_argument("--debug", "-debug",
                        action="store_true",
                        dest="debug",
                        help="Debug plots.",)

    parser.add_argument("--frames-to-plot", "-nframes", "--nframes",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        default=[200],
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
    cam = GradCAM(model, "block5_conv3")

    # get the input shape for cropping the candidates
    inp_shape = model.input_shape
    print("  Model loaded.")

    # load wave breaking candidates
    df = pd.read_csv(args.input[0])

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
    end = start + int(args.nframes[0])
    frames = frames[start:end]

    # set a palette
    vmin = float(args.vmin[0])
    palette = copy(plt.cm.magma)
    palette.set_under(alpha=0.0)

    # loop over frames
    print("\n  Looping over frames")
    dfs = []
    for k, fname in enumerate(frames):
        print("   -  processing frame {} of {}".format(k + 1, len(frames)))

        # open a new figure
        fig, ax = plt.subplots(figsize=(8, 8))

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
        _left = []
        _right = []
        _top = []
        _bottom = []
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
                            X = np.expand_dims(crop, axis=0)

                            pred = model.predict(X)

                            if pred > 0.5:
                                # compute the heat map
                                heatmap = cam.compute_heatmap(X, 1, eps=1e-20)
                                # heatmap[heatmap <= 200] = np.ma.masked
                                Zm = np.ma.masked_where(heatmap < vmin, heatmap)
                                # add to plot
                                ax.imshow(heatmap, zorder=20, alpha=0.5,
                                          cmap=palette, vmin=vmin, vmax=255,
                                          extent=[left, right, bottom, top])
                        except Exception:
                            # could not use this data for the love of god.
                            pass

        ax.imshow(frm, cmap="Greys_r")

        # annotations
        txt = "Frame {} of {}".format(frm_id, len(frames))
        ax.text(0.05, 0.95, txt, color="deepskyblue",
                va="top", zorder=100, transform=ax.transAxes,
                ha="left", fontsize=12,
                bbox=dict(boxstyle="square", ec="none", fc="0.1",
                          lw=1, alpha=0.7))

        # try to compute the ROI
        try:
            _, roi_rect, _ = compute_roi(roi, fname)
        except Exception:
            roi_rect = False
        if roi_rect:
            ax.add_patch(copy(roi_rect))

        ax.set_xlabel("$i$ $[pixel]$")
        ax.set_ylabel("$j$ $[pixel]$")

        fig.tight_layout()

        figname = str(frm_id).zfill(8) + "." + args.image_format[0]
        plt.savefig(os.path.join(temp_path, figname),
                    bbox_inches="tight", pad_inches=0.1,
                    dpi=200)
        plt.close()

    print("\nMy work is done!\n")
