# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   :
# POURPOSE : Track broken waves.
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : XX/XX/XXXX [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import os

import argparse

import numpy as np

import xarray as xr
import pandas as pd

from pathlib import Path

from skimage.io import imread, imsave
from skimage.util import img_as_ubyte

from glob import glob
from natsort import natsorted

from scipy.io import loadmat

import miniball
from sklearn.cluster import DBSCAN, OPTICS

from copy import copy

import seaborn as sns
import matplotlib as mpl
# uncoment the line below if running on datamour
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "#E9E9F1",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2


def compute_roi(roi, frame_path):
    """
    Compute  the region of interest (ROI) a and mask.

    Input can be either a list of coordinates or a dataframe.

    Note the the format expected by this funcion is the same as what
    matplotlib.patches.Rectangle expects and is in image coordinates.

    roi = [top_left_corner, top_right_corner, length, height]

    Parameters:
    ----------
    roi : list, pandas.DataFrame, bool
        either a list or a pandas dataframe.
    frame_path : str
        a valid path pointing to a image file


    Returns:
    -------
    roi_coords : list
        a list of coordinates

    rec_patch :  matplotlib.patches.Rectangle
        a rectangle instance of the ROI

    mask : np.array
        a image array with everything outside the ROI masked
    """

    # if it is a dataframe
    if isinstance(roi, pd.DataFrame):

        # select frame
        idx = int(os.path.basename(frame_path).split(".")[0])
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
                                      edgecolor="r",
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


def cluster(img, eps, min_samples, backend="OPTICS"):
    """Cluster group of pixels."""
    ipx, jpx = np.where(img)  # gets where img == 1
    X = np.vstack([ipx, jpx]).T

    if backend == "OPTICS":
        db = OPTICS(cluster_method="dbscan",
                    metric="euclidean",
                    eps=eps,
                    max_eps=eps,
                    min_samples=min_samples,
                    min_cluster_size=min_samples,
                    n_jobs=1,
                    algorithm="ball_tree").fit(X)
    elif backend == "DBSCAN":
        db = DBSCAN(eps=eps,
                    metric="euclidean",
                    min_samples=min_samples,
                    n_jobs=1,
                    algorithm="ball_tree").fit(X)
    else:
        raise ValueError("use either DBSCAN or OPICS.")
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # to dataframe
    df = pd.DataFrame(X, columns=["j", "i"])
    df["label"] = labels
    df = df[df["label"] >= 0]

    # get baricenters and radius
    l_center = []
    i_center = []
    j_center = []
    n_pixels = []
    radius = []
    for label, gdf in df.groupby("label"):
        center, r2 = miniball.get_bounding_ball(
            gdf[["i", "j"]].values.astype(float))
        i_center.append(center[0])
        j_center.append(center[1])
        l_center.append(label)
        n_pixels.append(len(gdf))
        radius.append(np.sqrt(r2))

    # to dataframe
    x = np.vstack([i_center, j_center, n_pixels, radius, l_center]).T
    columns = ["i", "j", "points", "radius", "label"]
    df = pd.DataFrame(x, columns=columns)

    return df


def main():
    """Call the main program."""
    pass


if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input folder .",)

    # input configuration file
    parser.add_argument("--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="frames",
                        required=True,
                        help="Input path with extracted frames.",)

    # input configuration file
    parser.add_argument("--surfaces", "-surfaces",
                        nargs=1,
                        action="store",
                        dest="surfaces",
                        default=[False],
                        required=False,
                        help="Input path with extracted surfaces.",)

    # ROI
    parser.add_argument("--region-of-interest", "-roi",
                        nargs=1,
                        action="store",
                        dest="roi",
                        required=True,
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

    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.frames[0]
    if os.path.isdir(inp):
        print("\n - Found the image data")

        frame_path = os.path.abspath(inp)
        frames = natsorted(glob(inp + "/*"))

        # build a grid for the image
        img = imread(frames[0])
        i_grid_img = np.linspace(0, img.shape[0], img.shape[0])
        j_grid_img = np.linspace(0, img.shape[1], img.shape[1])
        I, J = np.meshgrid(j_grid_img, i_grid_img)

    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # use surface information?
    has_surface = False
    if args.surfaces[0]:
        print("\n - Found the surfaces data")

        # load
        ds = xr.open_dataset(args.surfaces[0])

        has_surface = True

        # compute the scale
        print("  -- computing surfaces scale range")
        minima = []
        maxima = []
        for t, time in enumerate(ds["T"].values):
            tds = ds.isel(T=t)
            z = tds["Z"].values
            minima.append(np.nanmin(z))
            maxima.append(np.nanmax(z))
        zmin = np.nanmin(minima)
        zmax = np.nanmax(maxima)

    # handle region of interest
    if args.roi[0]:
        print("\n - Found the region of interest data")
        if os.path.isfile(args.roi[0]):
            roi = pd.read_csv(args.roi[0])
            # fill nans with the previous valid values
            roi = roi.fillna(method="backfill")
            # if len(roi) != len(frames):
            #     # if only the last frame is missing
            #     if abs(len(roi) - len(frames)) == 1:
            #         frames = frames[0:-1]
            #     else:
            #         raise ValueError("Number of rows in {} does not match. "
            #                          "number of frames in {}".format(
            #                              args.roi[0], inp))
        else:
            raise ValueError("ROI can be only a csv file in this case.")

    # deal with breaking data
    brk_path = args.input[0]
    brk_files = natsorted(glob(brk_path + "/*.mat"))

    # get the binary files
    bin_files = []
    bin_indexes = []
    for fname in brk_files:
        path = Path(fname)
        if path.name.startswith("Bin"):
            bin_files.append(fname)
            bin_indexes.append(int(path.name.split("_")[1]))
    bin_indexes = np.array(bin_indexes)

    # loop over frames
    print("\nLoopping over frames\n")
    for k, fname in enumerate(frames):
        print("  - processing frame {} of {}".format(k+1, len(frames)),
              end="\r")

        # load frame
        path = Path(fname)
        framenumber = int(path.name.strip(".png"))
        img = imread(fname)

        # compute ROI
        roi_coords, rec_patch, mask = compute_roi(roi, fname)

        # try to open the binary matrix
        has_foam = False
        idx_bin = np.where(bin_indexes == k)[0].tolist()
        if idx_bin:

            # load
            M = loadmat(bin_files[idx_bin[0]])
            foam = M["Foam_Mask"]

            # figure out the coverage qrea
            area = M["Z"]
            imin = int(area[0][0][0])
            imax = int(area[0][0][1])
            jmin = int(area[0][0][2])
            jmax = int(area[0][0][3])

            # coverage arrays
            [Ia, Ja] = np.meshgrid(np.arange(imin, imax + 1),
                                   np.arange(jmin, jmax + 1))
            Ia = Ia.T
            Ja = Ja.T

            # get the metric coordinates of the foam patches
            # brk = np.where(foam == 1)
            brk = Ia[foam[:, :] == 1], Ja[foam[:, :] == 1]

            # individual breaking pixels
            ibrk = brk[0]
            jbrk = brk[1]

            has_foam = True

        # plot
        fig, ax = plt.subplots(figsize=(8, 8))
        # axc = ax.twin(x)

        ax.imshow(img, cmap="Greys_r")
        if has_foam:
            ax.scatter(jbrk, ibrk, 10, marker=".", color="lawngreen",
                       zorder=20, alpha=0.5)

        ax.add_patch(copy(rec_patch))

        ax.set_xlabel("$i$ $[pixel]$")
        ax.set_ylabel("$j$ $[pixel]$")

        txt = "Frame {} of {}".format(k + 1, len(frames))
        ax.text(0.05, 0.05, txt, color="deepskyblue",
                va="bottom", zorder=100, transform=ax.transAxes,
                ha="left", fontsize=12,
                bbox=dict(boxstyle="square", ec="none", fc="0.1",
                          lw=1, alpha=0.7))

        # save plot
        fname = "tmp/{}.png".format(str(k).zfill(6))
        plt.savefig(fname, pad_inches=0.01, bbox_inches="tight", dpi=200)
        plt.close()

    print("\n\nMy work is done!\n")
