r"""
Plot the results from a wave breaking detecion algorithm

Usage:
-----
python plot_wave_breaking_detection_results.py --help

Example:
-------

python plot_wave_breaking_detection_results.py

Explanation:
-----------

-i : input from a compatible script.

--frames : input frames

--output-path : path to write temporary files and/or plots if in debug mode

--save-binary-masks : if parsed, will save the binary masks

SCRIPT   : plot_wave_breaking_detection_results.py
POURPOSE : plot the results from a wave breaking detecion algorithm
AUTHOR   : Caio Eadi Stringari
V2.0     : 16/04/2020 [Caio Stringari]

"""

import matplotlib as mpl
# mpl.use("agg")
import matplotlib.pyplot as plt

import os
import argparse

import itertools

from pathlib import Path

from glob import glob
from natsort import natsorted

import numpy as np

# pandas for I/O
import pandas as pd

from copy import copy

import re

import seaborn as sns
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# quite a pandas warning
pd.options.mode.chained_assignment = None


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
        input_text = frame_path
        res = re.search(regex, input_text)
        idx = int(res.group())

        roi = roi.loc[roi["frame"] == idx]
        roi = [int(roi["i"]), int(roi["j"]),
               int(roi["width"]), int(roi["height"])]

        # update mask and rectangle
        img = plt.imread(frame_path)
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
        img = plt.imread(frame_path)

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
            img = plt.imread(frame_path)
            mask = np.ones(img.shape)
            rec_patch = False

    # coordinates
    roi_coords = [roi[0], (roi[0] + roi[2]), roi[1],  (roi[1] + roi[3])]

    return roi_coords, rec_patch, mask


def plot(frmid, frm, df, breakers=False, roi=False, total_frames=-1, temp_path="tmp",):
    """
    Plot the results of the classification.

    Parameters:
    ----------
    frmid : int
        frame id. Usually a sequential number
    frm : np.ndarray
        a valid array for plt.imshow()
    df : pd.DataFrame
        dataframe with the detection/classification/clustering results
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
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(frm, cmap="Greys_r")

    if breakers:
        lines = []
        labels = []

        k = 0
        for g, gdf in df.groupby("wave_breaking_event"):

            j = 0
            for r, row in gdf.iterrows():
                # if radii are the same, draw a circle
                try:
                    r1 = row["ir"]
                    r2 = row["jr"]
                except Exception:
                    r1 = row["ir"]
                    r2 = r1
                if r1 == r2:
                    c = patches.Circle((row["ic"], row["jc"]), r1,
                                       edgecolor=row["color"],
                                       facecolor="none",
                                       linewidth=2)
                # if not, draw a ellipse
                else:
                    c = patches.Ellipse((row["ic"], row["jc"]),
                                        row["ir"]*2, row["jr"]*2,
                                        angle=row["theta_ij"],
                                        facecolor="none",
                                        edgecolor=row["color"],
                                        linewidth=2)
                ax.add_patch(copy(c))
                if j == 0:
                    lines.append(patches.Patch(color=row["color"]))
                    labels.append(
                        "Cluster: {}".format(row["wave_breaking_event"]))
                j += 1

            k += 1

        ax.legend(lines, labels, fontsize=14)

    txt = "Frame {} of {}".format(frmid, total_frames)
    ax.text(0.95, 0.05, txt, color="deepskyblue",
            va="top", zorder=100, transform=ax.transAxes,
            ha="right", fontsize=12,
            bbox=dict(boxstyle="square", ec="none", fc="0.1",
                      lw=1, alpha=0.7))

    # add ROI
    if roi:
        ax.add_patch(copy(roi))

    ax.set_xlabel("$i$ $[pixel]$")
    ax.set_ylabel("$j$ $[pixel]$")

    fig.tight_layout()
    figname = str(frmid).zfill(8)
    plt.savefig(os.path.join(temp_path, figname),
                bbox_inches="tight", pad_inches=0.1,
                dpi=200)
    plt.close()

    return fig, ax


def main():
    """Call the main program."""
    pass


if __name__ == "__main__":

    print("\nPlotting wave breaking, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
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

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["output/"],
                        required=False,
                        help="Output path.",)

    parser.add_argument("--region-of-interest", "-roi", "--roi",
                        nargs="*",
                        action="store",
                        dest="roi",
                        default=[False],
                        required=False,
                        help="Region of interest. Must be a file generated"
                             " with minmun_bounding_geometry.py",)

    parser.add_argument("--frames-to-plot", "-nframes", "--nframes",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        default=[2000],
                        help="How many frames to plot.",)

    args = parser.parse_args()

    # create the output path, if not present
    out_path = os.path.abspath(args.output[0])
    os.makedirs(out_path, exist_ok=True)

    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.frames[0]
    if os.path.isdir(inp):
        frames = natsorted(glob(inp + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # handle region of interest
    if args.roi[0]:
        try:
            roi = pd.read_csv(args.roi[0])
            # fill nans with the previous valid values
            roi = roi.fillna(method="backfill")

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
                # well, your data is just right
                pass
        except Exception:
            raise ValueError("Could not process region-of-interest file.")

    # load clustered data
    if args.input[0].endswith("csv"):
        df = pd.read_csv(args.input[0])
    else:
        try:
            df = pd.read_pickle(args.input[0])
        except Exception:
            raise IOError("Could not read file {}.".format(args.input[0]))

    # ckeck if all needed keys are present
    targets = ["ic", "jc", "ir", "frame", "wave_breaking_event"]
    for t in targets:
        if t not in df.keys():
            raise ValueError(
                "Key \"{}\" must be present in the data.".format(t))

    # map colours to unique events
    # k = 0
    colors = itertools.cycle(plt.cm.tab10(np.arange(1, 10, 1)))
    dfs = []
    n_events = df["wave_breaking_event"].unique()
    for g, gdf in df.groupby("wave_breaking_event"):
        color = mpl.colors.to_hex(next(colors), keep_alpha=True)
        gdf["color"] = color
        dfs.append(gdf["color"])
    df["color"] = pd.concat(dfs)

    # loop over frames
    print("  Looping over frames")
    dfs = []
    for k, fname in enumerate(frames):
        print("   -  processing frame {} of {}".format(k + 1, len(frames)),
              end="\r")

        # load the frame
        frm = plt.imread(fname)

        # locate the current frame in the breaking candidates dataframe
        frm_id = int(re.search(args.regex[0], fname).group())  # frame id

        df_frm = df.loc[df["frame"] == frm_id]

        # if there is a detection for that frame, apply the pipeline
        indexes = []  # valid dataframe indexes
        if not df_frm.empty:

            # try to compute the ROI
            try:
                _, roi_rect, _ = compute_roi(roi, fname, regex=args.regex[0])
            except Exception:
                roi_rect = False

            plot(frm_id, frm, breakers=True, df=df_frm, roi=roi_rect,
                 total_frames=len(frames), temp_path=out_path)

        else:

            # try to compute the ROI
            try:
                _, roi_rect, _ = compute_roi(roi, fname)
            except Exception:
                roi_rect = False

            plot(frm_id, frm, pd.DataFrame(), breakers=False, roi=roi_rect,
                 total_frames=len(frames), temp_path=out_path)

        if k >= int(args.nframes[0]):
            print("\n\n -- breaking the loop.")
            break

    print("\nMy work is done!\n")
