"""
Plot tracked waves.

PROGRAM   :
POURPOSE  :
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V2.0      : 30/09/2020 [Caio Stringari]
"""

import re
import os
import argparse

from tqdm import tqdm

from glob import glob
from natsort import natsorted

import numpy as np
import pandas as pd

from skimage.io import imread

from copy import copy

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from itertools import cycle

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "#E9E9F1",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2


if __name__ == '__main__':

    print("\nPlotting SORT results, please wait...\n")

    # argument parser
    parser = argparse.ArgumentParser()
    # input configuration file
    parser.add_argument("--tracks", "-tracks",
                        nargs=1,
                        action="store",
                        dest="tracks",
                        required=True,
                        help="Input path tracked waves. use track.py to get a valid file.",)

    parser.add_argument("--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with frames.",)

    parser.add_argument("--frames-to-process", "-nframes", "--nframes", "-N",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        default=[-1],
                        help="How many frames to process. Default is all.",)

    parser.add_argument("--from-frame", "-start", "--start",
                        nargs=1,
                        action="store",
                        dest="start",
                        default=[0],
                        help="In which frame to start processing. Default is 0.",)

    parser.add_argument("--regex", "-re",
                        nargs=1,
                        action="store",
                        dest="regex",
                        required=False,
                        default=["[0-9]{6,}"],
                        help="Regex to used when looking for files.",)

    parser.add_argument("--region-of-interest", "-roi", "-R",
                        nargs=4,
                        action="store",
                        dest="region_of_interest",
                        required=False,
                        default=[False, False, False, False],
                        help="Region of Interest. Format is top_left dx, dy.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="plot_path",
                        default=["tracks"],
                        required=False,
                        help="Output path.")

    args = parser.parse_args()

    # input images
    path = args.input[0]
    regex = args.regex[0]
    frames = natsorted(glob(path + "/*"))

    # create output path
    plot_path = args.plot_path[0]
    os.makedirs(plot_path, exist_ok=True)

    # read segmented pixels
    df = pd.read_csv(args.tracks[0])
    df = df.drop_duplicates()

    # select from which frame to start processing
    start = int(args.start[0])

    if int(args.nframes[0]) == -1:
        N = len(frames)
    else:
        N = int(args.nframes[0])
    total_frames = len(frames)
    frames = frames[start:start+N]

    # define region of interest
    roi = np.array(args.region_of_interest)
    if roi[0]:
        has_roi = True
        roi = np.array(args.region_of_interest).astype(int)
        roi_patch = mpatches.Rectangle((roi[0], roi[1]),
                                       roi[2], roi[3],
                                       linewidth=2,
                                       edgecolor="deepskyblue",
                                       facecolor="none",
                                       linestyle="--",
                                       zorder=20)
    else:
        has_roi = False

    pbar = tqdm(total=len(frames))

    # unique colors based on track
    colors = sns.color_palette("muted", 20).as_hex()
    color_cycle = cycle(colors)

    tracks = []
    track_colors = []
    for unique_track in df["track"].unique():
        track_colors.append(next(color_cycle))
        tracks.append(unique_track)
    colors = pd.DataFrame(tracks, columns=["track"])
    colors["color"] = track_colors


    # loop over images and get unique ids
    for i in range(len(frames)):

        # frame name
        f1 = frames[i]

        # ids
        res = re.search(regex, os.path.basename(f1))
        fid = int(res.group())

        # select in the dataframe
        df_frm = df.loc[df["frame"] == fid]

        # if there is data
        if not df_frm.empty:

            # read frame
            frame = imread(f1)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(frame, cmap="Greys_r")

            # plot tracks
            for track, tdf in df_frm.groupby("track"):

                minr = tdf["minr"].values
                minc = tdf["minc"].values
                dx = tdf["dx"].values
                dy = tdf["dy"].values
                color = colors.loc[colors["track"] == track]["color"].values[0]

                rect = mpatches.Rectangle((minc, minr), dx, dy, label="Wave {}".format(int(track)),
                                          fill=False, linewidth=1, edgecolor=color)
                ax.add_patch(rect)

            # region of interest
            if has_roi:
                ax.add_patch(copy(roi_patch))

            lg = ax.legend(loc=1, fontsize=10)
            lg.get_frame().set_color("w")
            lg.get_frame().set_alpha(0.75)

            ax.set_xlim(0, frame.shape[1])
            ax.set_ylim(frame.shape[0], 0)

            ax.set_xlabel(r"$i$ [pixel]")
            ax.set_ylabel(r"$j$ [pixel]")
            ax.set_aspect("equal")
            sns.despine(ax=ax)

            txt = "Frame {} of {}".format(str(fid).zfill(5),
                                          str(len(frames)).zfill(5))
            ax.text(0.01, 0.01, txt, color="deepskyblue",
                    va="bottom", zorder=100, transform=ax.transAxes,
                    ha="left", fontsize=12,
                    bbox=dict(boxstyle="square", ec="none", fc="0.1",
                              lw=1, alpha=0.7))

            plt.savefig(os.path.join(plot_path, os.path.basename(f1)),
                        dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()

        pbar.update()

    print("\n\nMy work is done!")
