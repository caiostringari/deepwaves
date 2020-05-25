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
import matplotlib as mpl
# mpl.use("Agg")

import os

import argparse
import numpy as np

from glob import glob
from natsort import natsorted

from skimage.io import imread, imsave
from skimage.util import img_as_ubyte

import xarray as xr
import pandas as pd

from copy import copy

from scipy.interpolate import griddata

# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "#E9E9F1",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['patch.edgecolor'] = "k"

import warnings
warnings.filterwarnings("ignore")


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
        img = imread(frame_path + ".png")
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


def interp(tds, mask, zmin, zmax):
    """Interp surfaces to pixel domain"""

    z = tds["Z"].values

    # scale
    scaled_z = (z - zmin) / (zmax - zmin)

    # load pixel coordinates
    ipx = tds["iR"].values
    jpx = tds["jR"].values

    # build the tree search
    X = np.vstack([ipx.flatten(), jpx.flatten(), scaled_z.flatten()]).T
    df = pd.DataFrame(X, columns=["i", "j", "z"])
    df = df.dropna()

    # interpolate
    try:
        dst = griddata(df[["i", "j"]].values,
                       df["z"].values,
                       (I, J), method="linear")
    except Exception:
        dst = np.zeros(mask.shape)

    return dst


def main():
    pass


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
                        help="Input with file detected data.",)

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

    # load detected breking events
    input = args.input[0]
    df = pd.read_csv(input)

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

    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.frames[0]
    if os.path.isdir(inp):
        print("\n - Found the image data")

        frame_path = os.path.abspath(inp)
        frames = natsorted(glob(inp + "/*"))

        if has_surface:
            # build a grid for the image
            img = imread(frames[0])
            i_grid_img = np.linspace(0, img.shape[0], img.shape[0])
            j_grid_img = np.linspace(0, img.shape[1], img.shape[1])
            I, J = np.meshgrid(j_grid_img, i_grid_img)

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
            raise

    # create the output path, if not present
    output = os.path.abspath(args.output[0])
    os.makedirs(output, exist_ok=True)
    os.makedirs(output + "/img/", exist_ok=True)
    if has_surface:
        os.makedirs(output + "/srf/", exist_ok=True)
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
            fname = os.path.join(frame_path,
                                 str(int(row["frame"])).zfill(8))
            img = imread(fname + ".png")

            # get ROI
            roi_coords, rec_patch, mask = compute_roi(roi, frame_path=fname)
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

            # get surface, if present
            if has_surface:

                i = int(row["source_frame"].split(".")[0])
                tds = ds.isel(T=i)
                surf = interp(tds, mask, zmin, zmax)

                # crop
                left = int(row["ic"] - (dx / 2))
                right = int(row["ic"] + (dx / 2))
                top = int(row["ic"] + (dy / 2))
                bottom = int(row["ic"] - (dy / 2))
                extent = [left, right, top, bottom]
                csurf = surf[bottom:top, left:right]

                if np.where(img_as_ubyte(csurf).flatten() == 0)[0].shape[0] > 0:
                    zeros.append(1)
                else:
                    zeros.append(0)
            else:
                zeros.append(0)

            # save cropped area
            fname = "{}/img/{}.png".format(output, str(k).zfill(6))
            imsave(fname, crop)
            if has_surface:
                fname = "{}/srf/{}.png".format(output, str(k).zfill(6))
                imsave(fname, img_as_ubyte(csurf))

            # open a new figure
            fig, ax = plt.subplots(figsize=(12, 10))

            # plot the image and detected instance
            ax.imshow(full_img, cmap="Greys_r", vmin=0, vmax=255)
            if has_surface:
                ax.imshow(surf, vmin=0, vmax=0.75, cmap="plasma", alpha=0.5)
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
                                    row["ir"].values[0]*2,
                                    row["jr"].values[0]*2,
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

            # txt = "Sample {} of {}".format(k, N)
            # ax.text(0.01, 0.01, txt, color="deepskyblue",
            #         va="bottom", zorder=100, transform=ax.transAxes,
            #         ha="left", fontsize=14,
            #         bbox=dict(boxstyle="square", ec="none", fc="0.1",
            #                   lw=1, alpha=0.7))

            # save plot
            fig.tight_layout()
            fname = "{}/plt/{}.svg".format(output, str(k).zfill(6))
            plt.savefig(fname, pad_inches=0.01, bbox_inches="tight", dpi=300)
            fname = "{}/plt/{}.png".format(output, str(k).zfill(6))
            plt.savefig(fname, pad_inches=0.01, bbox_inches="tight", dpi=300)
            plt.close()

            # output
            df_samples.append(row)
            samples.append(str(k).zfill(6))

        except Exception:
            # raise
            print("  - warning: found an error, ignoring this sample.")
        k += 1

    # save dataframe to file
    df_samples = pd.concat(df_samples)
    fname = "{}/labels.csv".format(output)
    df_samples["sample"] = samples
    df_samples["label"] = "0"
    if has_surface:
        df_samples["zeros"] = zeros
    df_samples.to_csv(fname, index=False)

    print("\n\nMy work is done!\n")
