"""
Extract detections using clustering + fitting ellipses.

PROGRAM   : extrac_detections_by_clustering.,py
POURPOSE  : extract detections for tracking
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

import numpy.linalg as la

from skimage.io import imread

from sklearn.cluster import DBSCAN

from numba import jit

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

pd.options.mode.chained_assignment = None


@jit(nopython=True)  # try to speed this up a bit
def mvee(points, tol=0.0001):
    """
    Finds the ellipse equation in center form (x-c).T * A * (x-c) = 1

    See:
    http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440

    Parameters:
    ----------
    points : np.ndarray
        Input points. It is an array N*M with N number of samples and M number
        of features (dimensions). In 2D it;s N*2.
    tol : float
        Tolerance for the algorithm. Defaults to 0.0001.

    Returns:
    -------
    A : np.ndarray
        Array with the ellipse parameters.
    C : np.ndarray
        Array with the centers of the ellipse.
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.outer(c, c)) / d
    return A, c


@jit(nopython=True)  # try to speed this up a bit
def get_ellipse_parameters(A):
    """
    Finds the ellipse paramters from A.

    See:
    http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440

    Parameters:
    ----------
    A : np.ndarray
        Use mvee to get the correct array.

    Returns:
    -------
    a, b : float
        major and minor axis of the ellipse
    C : np.ndarray
        Array with the centers of the ellipse.
    """
    # compute SVD
    U, D, V = la.svd(A)

    # x, y radii.
    rx, ry = 1. / np.sqrt(D)

    # Major and minor semi-axis of the ellipse.
    dx, dy = 2 * rx, 2 * ry
    a, b = max(dx, dy), min(dx, dy)

    # eccentricity
    e = np.sqrt(a ** 2 - b ** 2) / a

    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))

    # orientation angle (with respect to the x axis counterclockwise).
    theta = arccos if arcsin > 0. else -1. * arccos

    return a / 2, b / 2, theta, e


def main():
    """Call the main program."""

    # input images
    path = args.input[0]
    regex = args.regex[0]
    frames = natsorted(glob(path + "/*"))

    # ADD TO ARG PARSER
    eps = float(args.eps[0])
    min_samples = float(args.min_samples[0])

    # read segmented pixels
    df = pd.read_csv(args.pixels[0])

    plot = False
    if args.save_plots:
        plot = True
        plot_path = args.plot_path[0]
        os.makedirs(plot_path, exist_ok=True)

    # select from which frame to start processing
    start = int(args.start[0])

    if int(args.nframes[0]) == -1:
        N = len(frames)
    else:
        N = int(args.nframes[0])
    frames = frames[start:start+N]

    pbar = tqdm(total=len(frames))

    # output for SORT
    df_ellipses = []

    # --- time loop ---

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

            # clusterer
            # tqdm.write("DBSCAN")
            pbar.set_description("DBSCAN")
            clf = DBSCAN(metric="euclidean",
                         eps=eps,
                         min_samples=min_samples,
                         n_jobs=-1,
                         algorithm="ball_tree")
            clf.fit(df_frm[["i", "j"]])
            labels = clf.labels_

            # select only "insiders"
            df_frm["cluster"] = labels
            df_frm = df_frm.loc[df_frm["cluster"] >= 0]

            # fit the ellipse model for each cluster

            cols = ["frame", "cluster", "ic", "jc", "major_axis_length", "minor_axis_length", "angle", "area",
                    "eccentricity", "aspect_ratio", "minc", "minr", "dx", "dy"]
            df_ell = pd.DataFrame(columns=cols)

            k = 0
            pbar.set_description("MVEE")
            # tqdm.write("MVEE")
            for cl, gdf in df_frm.groupby("cluster"):

                # update the dataframe
                try:
                    # compute the minimun bounding ellipse
                    # note that the coordinates need to be reversed here somehow
                    A, c = mvee(gdf[["j", "i"]].values.astype(float), tol=0.1)
                    # centroid
                    xc, yc = c
                    # radius, angle and eccentricity
                    r1, r2, t, e = get_ellipse_parameters(A)
                    # area
                    a = np.pi * r1 * r2
                    # aspect ratio
                    r = r1/r2

                    # compute the boundary box of the ellipse
                    ux = r1 * np.cos(np.deg2rad(t))
                    uy = r1 * np.sin(np.deg2rad(t))
                    vx = r2 * np.cos(np.deg2rad(t) + np.pi/2)
                    vy = r2 * np.sin(np.deg2rad(t) + np.pi/2)

                    bbox_halfwidth = np.sqrt(ux*ux + vx*vx)
                    bbox_halfheight = np.sqrt(uy*uy + vy*vy)

                    bbox_ul_corner = (xc - bbox_halfwidth,
                                      yc - bbox_halfheight)

                    # bbox_br_corner = (xc + bbox_halfwidth,
                    #                   yc + bbox_halfheight)

                    # update
                    df_ell.at[k, :] = [gdf["frame"].values[0], cl, xc, yc,
                                       r1, r2, t, a, e, r,  # ellipse stuff
                                       bbox_ul_corner[0], bbox_ul_corner[1],  # bounding box stuff
                                       bbox_halfwidth*2, bbox_halfheight*2]

                    # append to global output
                    df_ellipses.append(df_ell)

                    k += 1

                # do not add anything to the dataframe
                except Exception:
                    pass

        # plot
        if plot:
            pbar.set_description("Plottting")
            im = imread(f1)
            fig, ax = plt.subplots()
            ax.imshow(im, cmap="Greys_r")

            # plot the ellipses
            colors = sns.color_palette("hls", len(df_ell))
            for i, row in df_ell.iterrows():
                c = mpatches.Ellipse((row["ic"], row["jc"]),
                                     row["major_axis_length"] * 2,
                                     row["minor_axis_length"] * 2,
                                     angle=row["angle"],
                                     facecolor="none",
                                     edgecolor=colors[i],
                                     linewidth=2)
                ax.add_artist(c)

            # add breaking pixels
            mk1 = np.zeros(im.shape)
            mk1[df_frm["i"].values, df_frm["j"].values] = 1
            mk1 = np.ma.masked_less(mk1, 1)
            ax.imshow(mk1, cmap=mpl.colors.ListedColormap("red"), alpha=1, zorder=10)

            fig.tight_layout()
            plt.savefig(os.path.join(plot_path, os.path.basename(f1)),
                        dpi=150, bbox_inches="tight", pad_inches=0.1)
            plt.close()
            # break

        pbar.update()

    dfo = pd.concat(df_ellipses)
    _df = dfo.drop_duplicates()
    _df.to_csv(args.output[0], index=False)


if __name__ == '__main__':

    print("\nExtracting detections for SORT, please wait...\n")

    # argument parser
    parser = argparse.ArgumentParser()

    # inputs
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
                        help="How many frames to process. Default is all (-1).",)
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

    # DBSCAN parameters
    parser.add_argument("--min-samples", "-nmin",
                        nargs=1,
                        action="store",
                        dest="min_samples",
                        default=[10],
                        required=False,
                        help="Minimum number of pixels to define a cluster.")
    parser.add_argument("--eps", "-eps",
                        nargs=1,
                        action="store",
                        dest="eps",
                        default=[2],
                        required=False,
                        help="Maximum distance allowed between pixels to form a cluster.")

    # output
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
                        help="Output file name (.csv).")

    args = parser.parse_args()

    main()

    print("\n\nMy work is done!")
