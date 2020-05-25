# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : naive_wave_breaking_detector.py
# POURPOSE : detect wave breaking using a "naive" local thresholding approach
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : caio.stringari@gmail.com
#
# V2.0     : 06/04/2020 [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
r"""
Detect wave breaking using a naive approach, i.e., by thresholding.

Usage:
-----
python naive_wave_breaking_detector.py --help

Example:
-------

python naive_wave_breaking_detector.py --debug \
                                       -i "input/folder/"  \
                                       -o "output" \
                                       --subtract-averages "average/folder" \
                                       --eps 10 \
                                       --min-samples 10 \
                                       --window-size 21 \
                                       --offset 10 \
                                       --region-of-interest "ROI.csv" \
                                       --temporary-path "tmp" \
                                       --fit-method "ellipse" \
                                       --nproc 4 \
                                       --save-binary-masks \
                                       --fill-regions \
                                       --block-shape 1024 1024

Explanation:
-----------

--debug : runs in debug mode, will save output plots

-i : input path with images

-o : output file name (see below for explanation)

--subtract-averages : input path with pre-computed average images.
                      use compute_average_image.py to get valid files

--eps : eps parameter for DBSCAN or OPTICS

--min-samples : min_samples parameter for DBSCAN or OPTICS

--window-size : window size for local_threshold

--offset : offset size for local_threshold

--region-of-interest : file with region of interest.
                       use minimun_bounding_geometry.py)

--temporary-path : path to write temporary files and/or plots if in debug mode

--fit-method : which geometry to fit to a detected cluster of bright points
               valid options are circle and ellipse.

--nproc 4 : number of processors to use if not in debub mode

--save-binary-masks : if parsed, will save the binary masks

--fill-regions : if parsed, will fill the regions inside a cluster.

--block-shape 1024 1024 : block shape to split the image into to avoid memory
                          errors

Other parameters:

--cluster-method : either DBSCAN or OPTICS. Defaults to DBSCAN.

--timeout : in parallel mode, kill a processes if taking longer than 120
            seconds per default.

Output:
------

The output CSV columns are organized as follows:

    ic : The i coordinate center of a cluster (image referential)
    jc : The j coordinate center of a cluster (image referential)
    pixels : Number of pixels in that cluster
    ir : Radius (or length if ellipse) of the cluster in pixels
    jr : Radius (or length if ellipse) of the cluster in pixels
    theta_ij : Angle of the cluster if fitted to an ellipse. Zero if circle
    cluster : Cluster ID
    block_i : Block index in the i-direction
    block_j : Block index in the j-direction
    block_i_left : Block start in the i-direction (image referential)
    block_i_right : Block end  in  the i-direction (image referential)
    block_j_top : Block end  in  the j-direction (image referential)
    block_j_bottom : Block start the j-direction (image referential)
    frame : sequential frame number

If "save_binary_mask" is passed will save the binary mask in the root folder
of this script.
"""

import matplotlib as mpl
# mpl.use("Agg")

import os
import shutil
import subprocess

import sys

import argparse
from glob import glob
from natsort import natsorted

import numpy as np

# parallel processing
from itertools import repeat
try:
    from pebble import ProcessPool
except Exception:
    ImportError("run pip install pebble.")
from concurrent.futures import TimeoutError

# decompression
import bz2

# image utils
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_local, threshold_otsu, threshold_sauvola
from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from skimage.util import view_as_blocks
from skimage.exposure import rescale_intensity

try:
    from pythreshold.global_th import min_err_threshold
    from pythreshold.global_th.entropy import kapur_threshold
    from pythreshold.utils import apply_threshold
except Exception:
    ImportError("run pip install pythreshold.")

try:
    import miniball
except Exception:
    ImportError("run pip install miniball.")

import numpy.linalg as la
from scipy.spatial import ConvexHull

# ML
from sklearn.utils import parallel_backend
from sklearn.cluster import DBSCAN, OPTICS

# pandas for I/O
import pandas as pd

# used only for debug
from copy import copy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# quite skimage warnings
import warnings
warnings.filterwarnings("ignore")

# This locks sklearn and makes sure it does not create more processes than
# what it"s being asked for.
parallel_backend("multiprocessing", n_jobs=1)


def mvee(points, tol=0.001):
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
    err = tol+1.0
    u = np.ones(N)/N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
        new_u = (1-step_size)*u
        new_u[jdx] += step_size
        err = la.norm(new_u-u)
        u = new_u
    c = np.dot(u, points)
    A = la.inv(np.dot(np.dot(points.T, np.diag(u)), points)
               - np.multiply.outer(c, c))/d
    return A, c


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
    rx, ry = 1./np.sqrt(D)

    # Major and minor semi-axis of the ellipse.
    dx, dy = 2 * rx, 2 * ry
    a, b = max(dx, dy), min(dx, dy)

    # eccentricity
    e = np.sqrt(a ** 2 - b ** 2) / a

    arcsin = -1. * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))

    # orientation angle (with respect to the x axis counterclockwise).
    theta = arccos if arcsin > 0. else -1. * arccos

    return a/2, b/2, theta, e


def split(a, n):
    """Split a list "a" into "n" parts."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def ensure_shape(img, block_shape):
    """
    Ensure that image shape is compatible with view_as_blocks.

    Block shape must be a power of 2 and will be coerced if not.
    Image will  be coerced to a shape that divides into block shape evenly.

    Parameters:
    ----------
    img : np.ndarray
        Input image.
    block_shape : list-like
        Block shape.

    Returns:
    -------
    img : np.ndarray
        Output image
    block_shape : list-like
        New block shape.
    """
    block_shape = np.array(block_shape)
    if not np.log2(block_shape[0]).is_integer():
        block_shape[0] = nextpow2(block_shape[0])
        print("     warning: block shape has been updated to a power of 2.")
    if not np.log2(block_shape[1]).is_integer():
        block_shape[1] = nextpow2(block_shape[1])
        print("     warning: block shape has been updated to a power of 2.")

    newsize = (nextpow2(img.shape[0]), nextpow2(img.shape[1]))

    ones = np.ones([newsize[0], newsize[1]], dtype=bool)
    ones[0:img.shape[0], 0:img.shape[1]] = img

    img = ones

    return img, block_shape


def task_done(future):
    """Check if a task is done or kill it if its taking too long to finish."""
    try:
        result = future.result()  # blocks until results are ready
    except TimeoutError as error:
        print("  -- process took longer than %d seconds" % error.args[1])
    except Exception as error:
        print("  process raised %s" % error)
        print(error.traceback)  # traceback of the function


def nextpow2(i):
    """
    Get the next power of 2 of a given number.

    Parameters:
    ----------
    i : int
        Any integer number.

    Returns:
    -------
    n : int
        Next power of 2 of i.
    """
    n = 1
    while n < i:
        n *= 2
    return n


def point_in_hull(point, hull, tolerance=1e-12):
    """Verify if a point is inside a convex Hull."""
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def compute_roi(roi, frame_path):
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


def cluster(img, eps, min_samples, backend="DBSCAN", fit_kind="circle",
            compute_convex_hull=False):
    """
    Cluster group of pixels.

    Parameters:
    ----------
    img : np.ndarray
        Input image. Must be binary.
    eps : float
        Maximum distance allowed to form a cluster.
    min_samples : int
        Minimum number of samples to form a cluster.
    backend : str
        Which backend to use for clustering. Default is DBSCAN.
    fit_kind : str
        What type of geometry to fir to the clusters. Default is circle.
    compute_convex_hull : bool
        If true, will compute the ConvexHull of the custers. Default is False.

    Returns:
    -------
    df : pd.DataFrame
        A dataframe with the clustering results.
    """
    ipx, jpx = np.where(img)  # gets where img == 1
    X = np.vstack([ipx, jpx]).T

    if len(X) <= min_samples:
        raise ValueError("Not enough samples to appy DBSCAN.")

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
        raise ValueError("Use either DBSCAN or OPTICS.")
    labels = db.labels_

    # to dataframe
    df = pd.DataFrame(X, columns=["j", "i"])
    df["cluster"] = labels
    df = df[df["cluster"] >= 0]

    # get centers and radii
    cluster = []
    i_center = []
    j_center = []
    n_pixels = []
    R1 = []
    R2 = []
    theta = []
    hulls = []
    for cl, gdf in df.groupby("cluster"):

        # fit a circle
        if fit_kind == "circle":
            c, r2 = miniball.get_bounding_ball(
                gdf[["i", "j"]].values.astype(float))
            xc, yc = c
            r1 = np.sqrt(r2)
            r2 = r1   # these are for ellipses only
            t = 0  # these are for ellipses only
        elif fit_kind == "ellipse":
            try:
                # compute the minmun bounding ellipse
                A, c = mvee(gdf[["i", "j"]].values.astype(float))
                # centroid
                xc, yc = c
                # radius, angle and eccentricity
                r1, r2, t, _ = get_ellipse_parameters(A)
            except Exception:
                # fall back to circle
                c, r2 = miniball.get_bounding_ball(
                    gdf[["i", "j"]].values.astype(float))
                xc, yc = c
                r1 = np.sqrt(r2)
                r2 = r1   # these are for ellipses only
                t = 0  # these are for ellipses only
        else:
            raise ValueError("Can only fit data to circles or ellipses.")
        # append to output
        i_center.append(xc)
        j_center.append(yc)
        cluster.append(cl)
        n_pixels.append(len(gdf))
        R1.append(r1)
        R2.append(r2)
        theta.append(t)

        if compute_convex_hull:
            hull = ConvexHull(gdf[["i", "j"]].values.astype(float))
            hulls.append(hull)

    # to dataframe
    x = np.vstack([i_center, j_center, n_pixels,
                   R1, R2, theta,
                   cluster]).T
    columns = ["ic", "jc", "pixels", "ir", "jr", "theta_ij", "cluster"]
    df = pd.DataFrame(x, columns=columns)

    return df, hulls


def detector(frame, average, path, eps, min_samples, win_size,
             offset, roi, debug, total_frames=False,
             cluster_kind="DBSCAN", fit_kind="circle",
             block_shape=(256, 256),
             save_binary_masks=False,
             THRESHOLD_METHOD="kapur", THRESHOLD_ONLY=False,
             FILL_CLUSTERS=False):
    """
    Detect whitecapping using a local thresholding approach.

    There is no output, this function will write to file.

    Parameters:
    ----------
    frame : str
        Full frame path
    averages : str
        Full average path.
        Use compute_average_image.py to obtain the averages.
    path : str
        Output path for the processed data.
    eps : foat
        The eps parameter for DBSCAN.
    min_samples : int
        The min_samples parameter for DBSCAN.
    win_size : int
        Size of the "convolution" box for local_threshold.
    offset : int
        Not exactly sure what this does. See skimage documentation.
    roi : list or pd.DataFrame
        Region of interest for processing.
        Use minmun_bounding_geometry.py to obtain a valid file.
    debug : bool
        Run in debug mode. will run in serial and plot outputs.
    total_frames: int
        Total number of frames. for ploting only
    cluster_kind : str
        Either DBSCAN or OPTICS.
    fit_kind : str
        What geometry to fit to a cluster. Can be either circle or ellipse.
    block_shape : tupple
        bBlock size for view_as_blocks. must be a power of 2
    save_binary_mask : bool
        If true, will save the binary image

    Returns:
    -------
        Nothing. Will write to file instead.
    """

    PID = os.getpid()
    print("  -- started processing frame:",
          os.path.basename(frame).split(".")[0], "PID:", PID)

    # ---- try the detection pipeline ----
    try:

        # load the images
        if frame.endswith("bz2"):

            # copy and decompress
            frame_nbr = int(os.path.basename(frame).split("_")[1])
            newframepath = "decomp/{}.tif.bz2".format(str(frame_nbr).zfill(8))
            shutil.copyfile(frame, newframepath)

            cmd = "bzip2 -d {}".format(newframepath)
            subprocess.run(cmd, shell=True)

            # now its a regular file
            wrkframe = newframepath.strip(".bz2")

        else:
            wrkframe = frame

        # read
        img = img_as_float(rgb2gray(imread(wrkframe)))

        # ---- deal with roi ----
        try:
            roi_coords, roi_rect, mask = compute_roi(roi, wrkframe)
        except Exception:
            print("  -- died because of ROI processing frame:",
                  os.path.basename(wrkframe).split(".")[0], "PID:", PID)
            return 0

        # read average image
        avg = img_as_float(rgb2gray(imread(average)))

        # remove average and mask where the intensity decreases
        dif = img - avg
        dif[dif < 0] = 0
        mask = rgb2gray(mask)

        # TODO: Verify effect of scalling instead of subtraction

        dif = img_as_ubyte(dif)
        img = img_as_ubyte(img)
        avg = img_as_ubyte(avg)

        # threshold
        if THRESHOLD_METHOD == "otsu":

            global_thresh = threshold_otsu((dif * mask))
            maxpx = (dif * mask).max()
            bin_img = (dif * mask) < (maxpx-global_thresh)

        elif THRESHOLD_METHOD == "sauvola":
            thresh_sauvola = threshold_sauvola(dif * mask,
                                               window_size=win_size)
            bin_img = dif > thresh_sauvola

        elif THRESHOLD_METHOD == "kapur":
            kptrx = kapur_threshold(dif)
            bin_img = apply_threshold(dif*mask, kptrx)
            bin_img = np.invert(bin_img)

        elif THRESHOLD_METHOD == "minerr":
            metrx = min_err_threshold(rescale_intensity(dif))
            bin_img = apply_threshold(dif*mask, metrx)
            bin_img = np.invert(bin_img)

        # default
        elif THRESHOLD_METHOD == "adaptative":
            local_thresh = threshold_local(dif * mask, win_size,
                                           offset=offset)
            bin_img = dif > local_thresh
        else:
            raise ValueError("THRESHOLD_METHOD is invalid.")

        # save binaty masks
        if save_binary_masks.lower() != "False".lower():
            fname = os.path.basename(wrkframe).split(".")[0]
            imsave("{}/{}".format(save_binary_masks, fname+".png"),
                   img_as_uint(np.invert(bin_img)))

        if THRESHOLD_ONLY:
            print("  -- finished processing frame:",
                  os.path.basename(wrkframe).split(".")[0], "PID:", PID)
            print("  -- only computed the threshold (as requested).")
            return 1

        # ensure the shape is right for processing as blocks
        bin_img, block_shape = ensure_shape(bin_img, block_shape)

        view = view_as_blocks(bin_img, tuple(block_shape.tolist()))

        if FILL_CLUSTERS:
            compute_hull = True
        else:
            compute_hull = False

        dfs = []  # store dbscan results
        Hulls = []  # only used if FILL_CLUSTERS is True
        i_offset = []  # only used if FILL_CLUSTERS is True
        j_offset = []  # only used if FILL_CLUSTERS is True
        for i in range(view.shape[0]):
            for j in range(view.shape[1]):

                # target block
                blk = view[i, j, :, :]

                # update indexes
                i1 = block_shape[0] * i
                i2 = i1 + block_shape[0]
                j1 = block_shape[1] * j
                j2 = j1 + block_shape[1]

                # try to group bright pixels
                try:
                    df, hulls = cluster(np.invert(blk), eps, min_samples,
                                        backend=cluster_kind,
                                        fit_kind=fit_kind,
                                        compute_convex_hull=compute_hull)
                    if isinstance(df, pd.DataFrame):
                        if not df.empty:

                            # fix offsets
                            # i, j need to swaped here, not sure why
                            df["ic"] = df["ic"] + j1
                            df["jc"] = df["jc"] + i1

                            # add info about the processing blocks
                            df["block_i"] = j
                            df["block_j"] = i
                            df["block_i_left"] = j1
                            df["block_i_right"] = j2
                            df["block_j_top"] = i1
                            df["block_j_bottom"] = i2

                            # append
                            dfs.append(df)

                        # if hulls,
                        if hulls:
                            for h in hulls:
                                Hulls.append(h)
                                i_offset.append(j1)
                                j_offset.append(i1)

                except Exception:
                    pass  # do nothing here
                    # it means that a block search failled

        # concatenate
        if dfs:
            df_dbscan = pd.concat(dfs)

            # add some extra information
            # df_dbscan["step"] = int(os.path.basename(frame).split(".")[0])
            df_dbscan["frame"] = os.path.basename(wrkframe).strip(".png")

            # write to file
            fname = os.path.basename(wrkframe).split(".")[0] + ".csv"
            df_dbscan.to_csv(os.path.join(path, fname), index=False)
        else:
            print("  -- died because of dbscan processing frame:",
                  os.path.basename(wrkframe).split(".")[0], "PID:", PID)
            return 0

        # if the convex hulls were computed, fill the regions defined by
        # each cluster. This is slow. Only do it if you really need this data
        if Hulls:
            i_inside = []
            j_inside = []
            for hull, ioff, joff in zip(Hulls, i_offset, j_offset):

                igrd = np.arange(hull.points[:, 0].min()-1,
                                 hull.points[:, 0].max()+1, 1)

                jgrd = np.arange(hull.points[:, 1].min()-1,
                                 hull.points[:, 1].max()+1, 1)

                X, Y = np.meshgrid(igrd, jgrd)

                for i, j in zip(X.flatten(), Y.flatten()):
                    if point_in_hull((i, j), hull):
                        i_inside.append(i + ioff)
                        j_inside.append(j + joff)

            # update binary image
            bin_img[np.array(j_inside).astype(int),
                    np.array(i_inside).astype(int)] = 0

        # debug plot
        if debug:
            fig, ax = plot(wrkframe, img, bin_img, block_shape, roi_rect,
                           df_dbscan, total_frames, fit_kind)

            # save to file
            # plt.show()
            fname = os.path.basename(wrkframe).split(".")[0] + ".png"
            plt.savefig(os.path.join(path, fname), dpi=200,
                        bbox_inches="tight", pad_inches=0.1)
            plt.close()

    except Exception:
        raise
        print("  -- died for some unknown reason processing frame:",
              os.path.basename(wrkframe).split(".")[0], "PID:", PID)

    print("  -- finished processing frame:",
          os.path.basename(wrkframe).split(".")[0], "PID:", PID)

    return 1


def plot(frame, img, bin_img, block_shape, roi_rect, df, total_frames,
         fit_kind):
    """
    Plot the results of the detection.

    Parameters:
    ----------
    frame : str
        Full frame path.
    img : np.ndarray
        Input image.
    bin_img : np.ndarray
        Binary image.
    block_shape : tupple
        Block size used for view_as_blocks.
    roi_rect : patches.Rectangle
        ROI instance.
    df : pd.DataFrame
        Clustering results.
    total_frames : int
        The total number of frames.
    fit_kind : str
        What geometry to fit to a cluster. Can be either circle or ellipse.

    Returns:
    -------
    fig, ax : matplotlib.pyplot.subplots
        The figure and axis.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # draw image
    ax.imshow(img, cmap="Greys_r", vmin=0, vmax=255)

    # plot the idinetified breaking pixels
    bin = np.invert(bin_img).astype(int)
    bin = np.ma.masked_less(bin, 1)
    binmap = mpl.colors.ListedColormap("lawngreen")
    ax.imshow(bin, cmap=binmap, alpha=0.5)

    # draw the processing blocks
    k = 0
    for i, gdf in df.groupby(["block_i", "block_j"]):
        color = plt.cm.tab20(k)
        for i, row in gdf.iterrows():
            c = patches.Rectangle((row["block_i_left"],
                                   row["block_j_top"]),
                                  block_shape[0], block_shape[1],
                                  facecolor="none",
                                  edgecolor=color,
                                  linewidth=2)
            ax.add_artist(c)
        k += 1

    # draw dbscan results
    k = 0
    for _, gdf in df.groupby(["block_i", "block_j"]):
        for i, row in gdf.iterrows():
            color = plt.cm.tab20(i)
            ax.scatter(row["ic"], row["jc"], s=80, marker="+",
                       linewidth=2, alpha=1, color=color)
            if fit_kind == "circle":
                c = patches.Circle((row["ic"], row["jc"]),
                                   row["ir"],
                                   facecolor="none",
                                   edgecolor=color,
                                   linewidth=2)
            elif fit_kind == "ellipse":
                c = patches.Ellipse((row["ic"], row["jc"]),
                                    row["ir"]*2, row["jr"]*2,
                                    angle=row["theta_ij"],
                                    facecolor="none",
                                    edgecolor=color,
                                    linewidth=2)
            else:
                raise ValueError("Can fit to circles or ellipses.")
            ax.add_artist(c)
        k += 1

    # draw roi
    if isinstance(roi_rect, patches.Rectangle):
        ax.add_patch(copy(roi_rect))

    # draw frame number
    L = len(str(total_frames))
    frame_number = str(int(
        os.path.basename(frame).split(".")[0])).zfill(L)
    txt = "Frame {} of {}".format(frame_number, total_frames)
    ax.text(0.01, 0.01, txt, color="deepskyblue",
            va="bottom", zorder=100, transform=ax.transAxes,
            ha="left", fontsize=14,
            bbox=dict(boxstyle="square", ec="none", fc="0.1",
                      lw=1, alpha=0.7))

    # axis
    height, width = img.shape
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel(r"$i$ $[pixel]$")
    ax.set_ylabel(r"$j$ $[pixel]$")
    sns.despine(ax=ax)

    return fig, ax


def main():
    """Call the main program."""
    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.input[0]
    if DIACAM:
        possible_frames = natsorted(glob(inp+"/*_02.tif.bz2"))
        frames = []
        for f in possible_frames:
            # skip mean frames
            if "Mean_Frame_" not in f:
                frames.append(f)
    else:
        if os.path.isdir(inp):
            frames = natsorted(glob(inp + "/*"))
        else:
            raise IOError("No such file or directory \"{}\"".format(inp))

    # load roi and verify if its a file
    if args.roi[0]:
        is_roi_file = os.path.isfile(args.roi[0])

    # create the output path, if not present
    temp_path = os.path.abspath(args.temp_path[0])
    os.makedirs(temp_path, exist_ok=True)

    os.makedirs("decomp", exist_ok=True)

    # create binary mask output
    if SAVE_BINARY_MASK.lower() != "False".lower():
        os.makedirs(SAVE_BINARY_MASK, exist_ok=True)

    # find and match frames and averages
    avgs = args.subtract_avg[0]
    if avgs:
        # get a list of files
        avgs = natsorted(glob(avgs + "/*"))
        if not avgs:
            raise IOError("Check your input folder with averaged images.")

        # associate each file with a unique frame
        n = len(avgs)
        _frame_chunks = list(split(frames, n))

        averages = []
        for i, chunk in enumerate(_frame_chunks):
            for _ in chunk:
                averages.append(avgs[i])
    else:
        averages = [None] * len(frames)

    # handle region of interest
    if args.roi[0]:
        if is_roi_file:
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
                averages = averages[0:mframes]
                roi = roi.iloc[0:mframes]
            else:
                pass
        else:
            roi = False
            raise ValueError("Could not process region-of-interest file.")
    else:
        roi = False

    # full call for this function is
    # local_detector(frame, average, path, eps, min_samples, win_size,
    #                offset, roi, debug, total_frames=False,
    #                cluster_kind="DBSCAN", fit_kind="circle",
    #                block_shape=(256, 256),
    #                save_binary_masks=False,
    #                THRESHOLD_METHOD="kapur", THRESHOLD_ONLY=False,
    #                FILL_CLUSTERS=False)

    # debug - or serial case
    if args.debug:
        print("  + Detection in debug/serial mode")
        # ---- detect in serial mode ----
        frame_counter = 0
        for frame, average in zip(frames, averages):
            detector(frame, average, temp_path,
                     EPS, MIN_SAMPLES,
                     WIN_SIZE, OFFSET, roi, True,
                     len(frames), CLUSTER_KIND, FIT_KIND,
                     BLOCK_SHAPE, SAVE_BINARY_MASK,
                     THRESHOLD_METHOD, THRESHOLD_ONLY, FILL_CLUSTERS)
            # break the loop if reach the plotting limit
            if frame_counter+1 >= int(args.nframes[0]):
                print("-- Breaking the loop at {}.".format(frame_counter+1))
                break
            frame_counter += 1

    else:
        print("  + Detection in pararell mode")
        # ---- detect in pararell ---
        fargs = zip(frames, averages, repeat(temp_path), repeat(EPS),
                    repeat(MIN_SAMPLES), repeat(WIN_SIZE), repeat(OFFSET),
                    repeat(roi), repeat(False), repeat(len(frames)),
                    repeat(CLUSTER_KIND), repeat(FIT_KIND),
                    repeat(BLOCK_SHAPE), repeat(SAVE_BINARY_MASK),
                    repeat(THRESHOLD_METHOD),
                    repeat(THRESHOLD_ONLY), repeat(FILL_CLUSTERS))
        with ProcessPool(max_workers=int(args.nproc[0]), max_tasks=99) as pool:
            for a in fargs:
                future = pool.schedule(detector, args=a,
                                       timeout=TIMEOUT)
                future.add_done_callback(task_done)

    if not THRESHOLD_ONLY:
        # merge all cdv files
        print("  + Merging outputs")
        dfs = []
        for fname in natsorted(glob(temp_path + "/*.csv")):
            dfs.append(pd.read_csv(fname))
        df = pd.concat(dfs)
        df.to_csv(args.output[0], index=False, chunksize=2**16)

    if not args.debug:
        shutil.rmtree(temp_path)

    # clean decompressed files
    shutil.rmtree("decomp")


if __name__ == "__main__":

    print("\nDetecting wave breaking, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with extracted frames.",)
    # DIACAM files?
    parser.add_argument("--diacam", "-diacam", "--DIACAM",
                        action="store_true",
                        dest="DIACAM",
                        help="If parsed, assume DIACAM structure.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["diferential_detector.csv"],
                        required=False,
                        help="Output file name.",)

    parser.add_argument("--subtract-averages",
                        nargs=1,
                        action="store",
                        dest="subtract_avg",
                        default=[False],
                        required=False,
                        help="Subtract average images from each frame."
                             "Must be path-like.",)

    parser.add_argument("--nproc", "-nproc",
                        nargs=1,
                        action="store",
                        dest="nproc",
                        default=[1],
                        required=False,
                        help="Number of processes to use.",)

    parser.add_argument("--cluster-method", "-cluster-method",
                        nargs=1,
                        action="store",
                        dest="cluster_kind",
                        default=["DBSCAN"],
                        required=False,
                        help="Either DBSCAN or OPTICS. Default is DBSCAN.",)

    parser.add_argument("--eps", "-eps",
                        nargs=1,
                        action="store",
                        dest="eps",
                        default=[10],
                        required=False,
                        help="DBSCAN eps parameter (pixels)",)

    parser.add_argument("--min-samples", "-min-samples",
                        nargs=1,
                        action="store",
                        dest="min_samples",
                        default=[5],
                        required=False,
                        help="DBSCAN min_samples parameter (pixels)",)

    parser.add_argument("--use-threshold-otsu",
                        nargs=1,
                        action="store",
                        dest="otsu",
                        default=["False"],
                        required=False,
                        help="If parsed as True will use OTSU method.",)

    parser.add_argument("--use-threshold-sauvola",
                        nargs=1,
                        action="store",
                        dest="sauvola",
                        default=["False"],
                        required=False,
                        help="If parsed as True will use Sauvola method.",)

    parser.add_argument("--use-threshold-kapur",
                        nargs=1,
                        action="store",
                        dest="kapur",
                        default=["False"],
                        required=False,
                        help="If parsed as True will use Kapur method.",)

    parser.add_argument("--use-threshold-minerr",
                        nargs=1,
                        action="store",
                        dest="minerr",
                        default=["False"],
                        required=False,
                        help="If parsed as True will use MinErr method.",)

    parser.add_argument("--window-size", "-window-size",
                        nargs=1,
                        action="store",
                        dest="window",
                        default=[11],
                        required=False,
                        help="Window size for local_threshold. Must be Odd.",)

    parser.add_argument("--offset", "-offset",
                        nargs=1,
                        action="store",
                        dest="offset",
                        default=[10],
                        required=False,
                        help="Offset for local_threshold.",)

    parser.add_argument("--region-of-interest", "-roi", "--roi",
                        nargs="*",
                        action="store",
                        dest="roi",
                        default=[False],
                        required=False,
                        help="Region of interest. Must be a file generated"
                             " with minmun_bounding_geometry.py",)

    parser.add_argument("--fit-method", "-fit-method",
                        nargs=1,
                        action="store",
                        dest="fitting_kind",
                        default=["ellipse"],
                        required=False,
                        help="Which geometry to use to fit the data."
                             " Either circle or ellipse. Defaults to ellipse",)

    parser.add_argument("--block-shape", "-block-shape",
                        nargs=2,
                        action="store",
                        dest="block_shape",
                        default=[256, 256],
                        required=False,
                        help="Size of the block to divide the image into."
                             " Must be a power of two. If not, will be cast.",)

    parser.add_argument("--save-binary-masks",
                        nargs=1,
                        action="store",
                        default=["False"],
                        dest="save_binary_mask",
                        required=False,
                        help="If parsed with an arguent, will save masks.",)

    parser.add_argument("--fill-regions",
                        action="store_true",
                        dest="fill_regions",
                        required=False,
                        help="Fill the regions occupied by a cluster.",)

    parser.add_argument("--temporary-path", "-temporary-path",
                        nargs=1,
                        action="store",
                        dest="temp_path",
                        default=["temp/"],
                        required=False,
                        help="Temporary folder name.",)

    parser.add_argument("--debug", "-debug",
                        action="store_true",
                        dest="debug",
                        required=False,
                        help="If parsed, will run in debug mode.",)

    parser.add_argument("--frames-to-plot", "-nframes", "--nframes",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        default=[200],
                        help="How many frames to plot.",)

    parser.add_argument("--threshold-only",
                        action="store_true",
                        dest="threshold_only",
                        help="Only compute the threshold and binary masks.",)

    parser.add_argument("--timeout",
                        action="store",
                        dest="timeout",
                        default=[120],
                        required=False,
                        help="Kill a process if taking more than 120 secs",)

    args = parser.parse_args()

    # Constants

    # which geometry to fit to wave breaking event candidates
    FIT_KIND = args.fitting_kind[0]

    # DBSCAN options
    CLUSTER_KIND = args.cluster_kind[0]
    EPS = float(args.eps[0])
    MIN_SAMPLES = int(args.min_samples[0])

    # deal with thresholding method
    THRESHOLD_METHOD = "adaptative"
    if args.otsu[0].lower() == "true":
        THRESHOLD_METHOD = "otsu"
    if args.sauvola[0].lower() == "true":
        THRESHOLD_METHOD = "sauvola"
    if args.kapur[0].lower() == "true":
        THRESHOLD_METHOD = "kapur"
    if args.minerr[0].lower() == "true":
        THRESHOLD_METHOD = "minerr"

    print("\n -- Thresholding method is: {}\n".format(THRESHOLD_METHOD))

    # window and offset for local threshold
    WIN_SIZE = int(args.window[0])
    OFFSET = int(args.offset[0])

    BLOCK_SHAPE = [int(args.block_shape[0]), int(args.block_shape[0])]

    TIMEOUT = int(args.timeout[0])

    # fill clusters if using local_threshold, this is slow!
    FILL_CLUSTERS = args.fill_regions

    # save binary masks if true
    SAVE_BINARY_MASK = args.save_binary_mask[0]

    # THRESHOLD ONLY
    THRESHOLD_ONLY = args.threshold_only

    # DIACAM file structure?
    DIACAM = args.DIACAM

    # call the main program
    main()

    print("\nMy work is done!\n")
