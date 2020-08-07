# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : prepare_data_for_classifier.py
# POURPOSE : extract data for a machine learning breaking classifier.
#            use naive_wave_breaking_detector.py to get a valid input file.
# AUTHOR   : Caio Eadi Stringari
#
# V1.0     : 04/05/2020 [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import warnings
import matplotlib as mpl
# mpl.use("Agg")

import os

import re

import argparse
import numpy as np

from glob import glob
from natsort import natsorted

from skimage.io import imread, imsave
# from skimage.util import img_as_ubyte

import pandas as pd

from copy import copy

# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "#E9E9F1",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['patch.edgecolor'] = "k"

warnings.filterwarnings("ignore")


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


if __name__ == "__main__":

    print("\nExtracting data for the classifier, please wait...")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input with file naive data.",)

    parser.add_argument("--target-class", "-target-class",
                        nargs=1,
                        action="store",
                        dest="target_class",
                        default=[-1],
                        required=False,
                        help="Target a label if the datata has been previously classified.")

    # input configuration file
    parser.add_argument("--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="frames",
                        required=True,
                        help="Input path with extracted frames.",)

    parser.add_argument("--regex", "-re", "-regex",
                        nargs=1,
                        action="store",
                        dest="regex",
                        required=False,
                        default=["[0-9]{6,}"],
                        help="Regex to search for frames. Default is [0-9]{6,}.",)

    # ROI
    parser.add_argument("--region-of-interest", "-roi",
                        nargs=1,
                        action="store",
                        dest="roi",
                        default=[False],
                        required=False,
                        help="Region of interest. Must be a csv file.",)

    # samples
    parser.add_argument("--N", "-n", "-N", "--samples",
                        nargs=1,
                        action="store",
                        dest="N",
                        required=False,
                        default=[1000],
                        help="Number of samples to extract.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["classifier/"],
                        required=False,
                        help="Output path.",)

    parser.add_argument("--size", "-size",
                        nargs=2,
                        action="store",
                        dest="size",
                        default=[256, 256],
                        required=False,
                        help="Image size.",)

    args = parser.parse_args()

    # load detected breking events
    input = args.input[0]
    df = pd.read_csv(input)

    # select a class, if asked
    target = int(args.target_class[0])
    if target > 0:
        try:
            df = df.loc[df["class"] == target]
        except Exception:
            raise ValueError("Key class is not in dataframe.")

    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.frames[0]
    if os.path.isdir(inp):
        print("\n - Found the image data")
        frame_path = os.path.abspath(inp)
        frames = natsorted(glob(inp + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # get frame names
    frame_str_list = []
    for frame in frames:
        res = re.search(args.regex[0], os.path.basename(frame))
        frame_str_list.append(res.group())
    df_frm = pd.DataFrame()
    df_frm["frame"] = frames
    df_frm["string"] = frame_str_list
    pad = len(frame_str_list[0])

    # handle region of interest
    # load roi and verify if its a file
    if args.roi[0]:
        is_roi_file = os.path.isfile(args.roi[0])
    if args.roi[0]:
        if is_roi_file:
            roi = pd.read_csv(args.roi[0])
            # fill nans with the previous/next valid values
            roi = roi.fillna(method="bfill")
            roi = roi.fillna(method="ffill")
        else:
            roi = False
            raise ValueError("Could not process region-of-interest file.")
    else:
        roi = False

    # create the output path, if not present
    output = os.path.abspath(args.output[0])
    os.makedirs(output, exist_ok=True)
    os.makedirs(output + "/img/", exist_ok=True)
    os.makedirs(output + "/plt/", exist_ok=True)

    # verify if "cluster", "ic", "jc", R1, "source_frame" are in df
    keys = df.keys()
    targets = ["ic", "jc", "cluster", "ir", "frame"]
    for t in targets:
        if t not in keys:
            raise ValueError(
                "Input data must have a column named \'{}\'".format(t))

    # number of samples
    N = int(args.N[0])

    # image size
    dx = int(args.size[1])
    dy = int(args.size[1])

    k = 0
    df_samples = []
    samples = []
    zeros = []
    while k < N:
        try:
            print("  - processing frame {} of {}".format(k + 1, N))

            row = df.sample()

            # load frame
            frame_number = str(int(row["frame"])).zfill(pad)
            frame_path = df_frm.loc[df_frm["string"] == frame_number]["frame"].values[0]
            img = imread(frame_path)

            # get ROI
            roi_coords, rec_patch, mask = compute_roi(roi, frame_path=frame_path)
            full_img = copy(img)

            # crop image
            try:
                height, width, c = full_img.shape
            except Exception:
                height, width = full_img.shape

            left = int(row["ic"] - (dx / 2))
            if left < 0:
                left = 0
            right = int(row["ic"] + (dx / 2))
            if right > width:
                right = width
            top = int(row["jc"] + (dy / 2))
            if top < 0:
                top = 0
            if top > height:
                top = height
            bottom = int(row["jc"] - (dy / 2))
            if bottom < 0:
                bottom = 0
            extent = [left, right, top, bottom]
            crop = img[bottom:top, left:right]

            # save cropped area
            fname = "{}/img/{}.png".format(output, str(k).zfill(6))
            imsave(fname, crop)

            # open a new figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # plot the image and detected instance
            ax.imshow(full_img, cmap="Greys_r", vmin=0, vmax=255)
            ax.scatter(row["ic"], row["jc"], 80, marker="+",
                       linewidth=2, color="r", alpha=1)
            ax.add_patch(copy(rec_patch))

            # plot inset
            axins = ax.inset_axes([0.7, 0.05, 0.3, 0.3])
            axins.imshow(crop, cmap="Greys_r", vmin=0, vmax=255,
                         extent=extent)

            # if radii are the same, draw a circle
            try:
                r1 = row["ir"].values[0]
                r2 = row["jr"].values[0]
            except Exception:
                r1 = row["ir"].values[0]
                r2 = r1
            if r1 == r2:
                c = patches.Circle((row["ic"].values[0],
                                    row["jc"].values[0]), r1,
                                   edgecolor="r", facecolor="None")
            # if not, draw a ellipse
            else:
                c = patches.Ellipse((row["ic"].values[0],
                                     row["jc"].values[0]),
                                    row["ir"].values[0] * 2,
                                    row["jr"].values[0] * 2,
                                    angle=row["theta_ij"].values[0],
                                    facecolor='none',
                                    edgecolor="r",
                                    linewidth=2)
            ax.add_patch(copy(c))
            axins.add_patch(copy(c))

            # draw the connection line
            rp, cl = ax.indicate_inset_zoom(axins, linewidth=2,
                                            edgecolor="lawngreen",
                                            facecolor="none", alpha=1)
            for line in cl:
                line.set_color("w")
                line.set_linestyle("--")

            # set extent
            axins.set_xlim(left, right)
            axins.set_ylim(top, bottom)

            # turn axis off
            axins.set_xticks([])
            axins.set_yticks([])
            for spine in axins.spines.values():
                spine.set_edgecolor("lawngreen")
                spine.set_linewidth(2)
            ax.set_xlabel("$i$ [pixel]")
            ax.set_ylabel("$j$ [pixel]")

            txt = "Sample {} of {}".format(k, N)
            ax.text(0.01, 0.01, txt, color="deepskyblue",
                    va="bottom", zorder=100, transform=ax.transAxes,
                    ha="left", fontsize=14,
                    bbox=dict(boxstyle="square", ec="none", fc="0.1",
                              lw=1, alpha=0.7))

            # save plot
            fig.tight_layout()
            # fname = "{}/plt/{}.svg".format(output, str(k).zfill(6))
            # plt.savefig(fname, pad_inches=0.01, bbox_inches="tight", dpi=300)
            fname = "{}/plt/{}.png".format(output, str(k).zfill(6))
            plt.savefig(fname, pad_inches=0.01, bbox_inches="tight", dpi=300)
            plt.close()

            # output
            df_samples.append(row)
            samples.append(str(k).zfill(6))

        except Exception:
            # raise
            print("  - warning: found an error, ignoring this sample.")
            raise
        k += 1

    # save dataframe to file
    df_samples = pd.concat(df_samples)
    fname = "{}/labels.csv".format(output)
    df_samples["sample"] = samples
    df_samples["label"] = target
    df_samples.to_csv(fname, index=False)

    print("\n\nMy work is done!\n")
