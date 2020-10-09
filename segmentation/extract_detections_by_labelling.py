"""
Extract detections using skimage label and regionprops functions.

Use this only if extrac_detections_by_clustering.py does not work for you.

PROGRAM   : extrac_detections_by_labelling
POURPOSE  : extract bounding boxes from segmented pixels
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
from skimage.measure import label, regionprops

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


if __name__ == '__main__':

    print("\nExtracting detections for SORT, please wait...\n")

    # argument parser
    parser = argparse.ArgumentParser()
    # input configuration file
    parser.add_argument("--input", "-i", "--frames", "-frames",
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

    parser.add_argument("--pixels", "-p", "--segmentation",
                        nargs=1,
                        action="store",
                        dest="pixels",
                        required=True,
                        help="DataFrame with selected pixel information."
                             "Use predict.py to get a valid file.",)

    parser.add_argument("--min-area", "-min-area",
                        nargs=1,
                        action="store",
                        dest="min_area",
                        default=[1000],
                        required=False,
                        help="Minimum area in pixels to define a bounding box.")
    parser.add_argument("--connectivity", "-conn",
                        nargs=1,
                        action="store",
                        dest="connectivity",
                        default=[1],
                        required=False,
                        help="Connectivity type. Default is 1.")

    parser.add_argument("--save-plots", "--plot",
                        action="store_true",
                        dest="save_plots",
                        required=False,
                        help="Save processed images. Will slow down the code.")
    parser.add_argument("--plot-path",
                        nargs=1,
                        action="store",
                        dest="plot_path",
                        default=["plot"],
                        required=False,
                        help="Save processed images. Will slow down the code.")
    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output file name.")

    args = parser.parse_args()

    # input images
    path = args.input[0]
    regex = args.regex[0]
    frames = natsorted(glob(path + "/*"))

    # read segmented pixels
    df = pd.read_csv(args.pixels[0])

    # minimum area
    min_area = int(args.min_area[0])

    # connectivity
    connectivity = int(args.connectivity[0])

    # output
    f = open(args.output[0], "w")

    # write header
    f.write("frame,none,minc,minr,dx,dy,perimeter,area,major_axis_length,minor_axis_length\n")

    plot = False
    if args.save_plots:
        plot = True
        plot_path = args.plot_path[0]
        os.makedirs(plot_path, exist_ok=True)
        colors = sns.color_palette("tab20", 32)

    # select from which frame to start processing
    start = int(args.start[0])

    if int(args.nframes[0]) == -1:
        N = len(frames)
    else:
        N = int(args.nframes[0])
    total_frames = len(frames)
    frames = frames[start:start+N]

    pbar = tqdm(total=len(frames))

    # loop over images and get unique ids
    for i in range(len(frames)):

        # frame name
        f1 = frames[i]

        # ids
        res = re.search(regex, os.path.basename(f1))
        fmrid = int(res.group())

        # select in the dataframe
        df_frm = df.loc[df["frame"] == fmrid]

        # if there is data
        if not df_frm.empty:

            # read frame
            frame = imread(f1)

            # build the mask
            mk1 = np.zeros(frame.shape)
            mk1[df_frm["i"].values, df_frm["j"].values] = 1

            # lalbe the mask
            lbl = label(mk1, connectivity=connectivity)

            # get region properties
            bboxes = []
            perimiter = []
            area = []
            major_axis_length = []
            minor_axis_length = []
            props = regionprops(lbl)
            for prop in props:
                if prop.area >= min_area:
                    bboxes.append(prop.bbox)
                    perimiter.append(prop.perimeter)
                    area.append(prop.area)
                    major_axis_length.append(prop.major_axis_length)
                    minor_axis_length.append(prop.minor_axis_length)

            # loop over detections
            if bboxes:
                for box, pr, ar, major, minor in zip(bboxes, perimiter, area, major_axis_length, minor_axis_length):
                    minr, minc, maxr, maxc = box
                    txt = "{},-1,{},{},{},{},{},{},{},{}".format(fmrid, minc, minr, maxc - minc, maxr - minr, pr, ar, major, minor)
                    f.write(txt + "\n")

            if plot:
                fig, ax = plt.subplots()
                ax.imshow(frame, cmap="Greys_r")
                if bboxes:
                    for k, box in enumerate(bboxes):
                        minr, minc, maxr, maxc = box
                        color = sns.color_palette("hls", len(bboxes))[k]
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                  fill=False, linewidth=1, edgecolor=color)
                    ax.add_patch(rect)

                # add breaking pixels
                mk1 = np.ma.masked_less(mk1, 1)
                ax.imshow(mk1, cmap=mpl.colors.ListedColormap("red"), alpha=1, zorder=10)

                fig.tight_layout()
                plt.savefig(os.path.join(plot_path, os.path.basename(f1)),
                            dpi=150, bbox_inches="tight", pad_inches=0.1)
                plt.close()

        pbar.update()
        # break

    # close the file
    f.close()

    print("\n\nMy work is done!")
