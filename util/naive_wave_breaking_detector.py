# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : naive_wave_breaking_detector.py
# POURPOSE : detect wave breaking using a "naive" local thresholding approach
# AUTHOR   : Caio Eadi Stringari
# V2.0     : 06/04/2020 [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
r"""
Detect wave breaking using a naive approach, i.e., by thresholding
and clustering.

Usage:
-----
python naive_wave_breaking_detector.py --help

Example:
-------
python naive_wave_breaking_detector.py --debug \
                                       -i "input/folder/"  \
                                       -o "output.csv" \
                                       --subtract-averages "average/folder" \
                                       --cluster "dbscan" 10 10
                                       --threshold "adaptative" 11 10,
                                       --region-of-interest "file.csv" \
                                       --temporary-path "tmp" \
                                       --fit-method "ellipse" \
                                       --nproc 4 \
                                       --block-shape 1024 1024

--debug : runs in debug mode, will use only 1 processor and save output plots

-i : input path with images

-o : output file name (see below for explanation)

--subtract-averages : input path with pre-computed average images.
                      use compute_average_image.py to get valid files

--cluster : cluster method and parameters. Only DBSCAN is functional.

--threshold : which thresholding method to use. Default is adaptative which
              requires the window size and offset. Valid options are: otsu,
              entropy, constant, and file.

--region-of-interest : file with region of interest. use minimun_bounding_geometry.py
                       to get a valid file.

--temporary-path : path to write temporary files and/or plots if in debug mode

--fit-method : which geometry to fit to a detected cluster of bright points
               valid options are circle and ellipse.

--nproc 4 : number of processors to use if not in debub mode

--block-shape 1024 1024 : block shape to split the image into to avoid memory
                          errors

--frames-to-plot : number of frames to plot if in debug mode.


Output:
------

The output csv columns are organized as follows:

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
"""
import os
import matplotlib as mpl
if os.name == 'posix' and "DISPLAY" not in os.environ:
    mpl.use('Agg')

import shutil

import argparse
from glob import glob
from natsort import natsorted

import numpy as np

# regular expressions =(
import re

# parallel processing
from itertools import repeat
try:
    from pebble import ProcessPool
except Exception:
    ImportError("run pip install pebble.")
from concurrent.futures import TimeoutError

# image utils
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_float, img_as_ubyte
from skimage.util import view_as_blocks
from skimage.filters import threshold_local
try:
    from pythreshold.global_th import otsu_threshold
    from pythreshold.global_th.entropy import kapur_threshold
    from pythreshold.utils import apply_threshold
except Exception:
    ImportError("run pip install pythreshold.")

try:
    import miniball
except Exception:
    ImportError("run pip install miniball.")

import numpy.linalg as la

# ML
from sklearn.utils import parallel_backend
from sklearn.cluster import DBSCAN, OPTICS

try:
    import hdbscan
except Exception:
    ImportError("run pip install hdbscan.")

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
               - np.multiply.outer(c, c)) / d
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


def cluster(img, eps, min_samples, backend="dbscan", nthreads=2,
            fit_kind="circle"):
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

    Returns:
    -------
    df : pd.DataFrame
        A dataframe with the clustering results.
    """
    ipx, jpx = np.where(img)  # gets where img == 1
    X = np.vstack([ipx, jpx]).T

    if len(X) > min_samples:
        if backend.lower() == "optics":
            db = OPTICS(cluster_method="dbscan",
                        metric="euclidean",
                        eps=eps,
                        max_eps=eps,
                        min_samples=min_samples,
                        min_cluster_size=min_samples,
                        n_jobs=nthreads,
                        algorithm="ball_tree").fit(X)
            labels = db.labels_
        elif backend.lower() == "hdbscan":
            db = hdbscan.HDBSCAN(min_cluster_size=int(min_samples),
                                 metric="euclidean",
                                 allow_single_cluster=True,
                                 core_dist_n_jobs=nthreads)
            labels = db.fit_predict(X)
        elif backend.lower() == "dbscan":
            db = DBSCAN(eps=eps,
                        metric="euclidean",
                        min_samples=min_samples,
                        n_jobs=nthreads,
                        algorithm="ball_tree").fit(X)
            labels = db.labels_
        else:
            raise ValueError("Use either DBSCAN or OPTICS.")

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

        # to dataframe
        x = np.vstack([i_center, j_center, n_pixels,
                       R1, R2, theta,
                       cluster]).T
        columns = ["ic", "jc", "pixels", "ir", "jr", "theta_ij", "cluster"]
        df = pd.DataFrame(x, columns=columns)

        return df

    else:
        return pd.DataFrame()


def detector(frame, average, roi, output, cluster_pars=["dbscan", 10, 10],
             threshold_pars=["otsu"],
             total_frames=False, fit_kind="circle",
             block_shape=(256, 256), nthreads=2, regex="[0-9]{6,}", debug=False):
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
    roi : list or pd.DataFrame
        Region of interest for processing.
        Use minmun_bounding_geometry.py to obtain a valid file.
    output : str
        Output path for the processed data.
    cluster_pars : list
        List of parameters for clustering.
    threshold_pars : list
        List of parameters for thresholding.
    total_frames: int
        Total number of frames. for plotting only
    fit_kind : str
        What geometry to fit to a cluster. Can be either circle or ellipse.
    block_shape : tupple
        Block size for view_as_blocks. must be a power of 2
    regex : string
        Regex to get the sequential image number "[0-9]{6,}".
    debug : bool
        Run in debug mode. will run in serial and plot outputs.


    Returns:
    -------
        Nothing. Will write to file instead.
    """
    # try to figure out frame number and process ID
    PID = os.getpid()
    frmid = int(re.search(regex, os.path.basename(frame)).group())
    print("  -- started processing frame", frmid, "of", total_frames, "with PID", PID)

    # ---- try the detection pipeline ----
    try:

        # read
        img = img_as_float(rgb2gray(imread(frame)))

        # ---- deal with roi ----
        try:
            roi_coords, roi_rect, mask = compute_roi(roi, frame, regex=regex)
        except Exception:
            print("   -- died because of Region of Interest processing frame", frmid)
            return 0

        # try to read average image
        try:
            avg = img_as_float(rgb2gray(imread(average)))
        except Exception:
            avg = np.zeros(img.shape)

        # remove average and mask where the intensity decreases
        dif = img - avg
        dif[dif < 0] = 0
        mask = rgb2gray(mask)

        # TODO: Verify effect of scalling instead of subtraction

        dif = img_as_ubyte(dif)
        img = img_as_ubyte(img)
        avg = img_as_ubyte(avg)

        # threshold
        if threshold_pars[0] == "otsu":
            trx = otsu_threshold(dif)
            bin_img = apply_threshold(dif * mask, trx)
            bin_img = np.invert(bin_img)

        elif threshold_pars[0] == "entropy":
            kptrx = kapur_threshold(dif)
            bin_img = apply_threshold(dif * mask, kptrx)
            bin_img = np.invert(bin_img)

        elif threshold_pars[0] == "adaptative":
            local_thresh = threshold_local(dif * mask, threshold_pars[1],
                                           offset=threshold_pars[2])
            bin_img = dif > local_thresh

        elif threshold_pars[0] == "constant":
            bin_img = apply_threshold(dif * mask, int(threshold_pars[1]))
            bin_img = np.invert(bin_img)

        elif threshold_pars[0] == "file":
            trxdf = pd.read_csv(threshold_pars[1])
            nearest = trxdf["frame"].values[np.argmin(np.abs(frmid-trxdf["frame"].values))]
            bin_img = apply_threshold(dif * mask, trxdf.iloc[nearest]["threshold"])
            bin_img = np.invert(bin_img)

        else:
            raise ValueError("Fatal: could not deal with thresholding method.")

        # ensure the shape is right for processing as blocks
        bin_img, block_shape = ensure_shape(bin_img, block_shape)

        view = view_as_blocks(bin_img, tuple(block_shape.tolist()))

        # outputs
        dfs = []  # store dbscan results

        # loop over image blocks
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
                    df = cluster(np.invert(blk),
                                 cluster_pars[1],
                                 cluster_pars[2],
                                 backend=cluster_pars[0],
                                 nthreads=nthreads,
                                 fit_kind=fit_kind)
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

                except Exception:
                    raise
                    pass  # do nothing here
                    # it means that a block search failled

        # concatenate
        if dfs:
            df_dbscan = pd.concat(dfs)

            # add some extra information
            # df_dbscan["step"] = int(os.path.basename(frame).split(".")[0])
            df_dbscan["frame"] = frmid

            # write to file
            fname = str(frmid).zfill(8) + ".csv"
            df_dbscan.to_csv(os.path.join(output, fname), index=False)
        else:
            print("   -- no clusters were found processing frame", frmid)
            df_dbscan = pd.DataFrame()
            return 0

        # debug plot
        if debug:
            fig, ax = plot(frmid, img, bin_img, block_shape, roi_rect,
                           df_dbscan, total_frames, fit_kind)

            # save to file
            fname = str(frmid).zfill(8) + ".png"
            plt.savefig(os.path.join(output, fname), dpi=150,
                        bbox_inches="tight", pad_inches=0.1)
            # plt.show()
            plt.close()

    except Exception:
        raise
        print("   -- died for some unknown reason processing frame", frmid)

    print("   -- finished processing frame", frmid)

    return 1


def plot(frmid, img, bin_img, block_shape, roi_rect, df, total_frames,
         fit_kind):
    """
    Plot the results of the detection.

    Parameters:
    ----------
    frmid : str
        frame sequential number.
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
    binmap = mpl.colors.ListedColormap("red")
    ax.imshow(bin, cmap=binmap, alpha=1, zorder=10)

    # draw the processing blocks
    k = 0
    if not df.empty:
        for i, gdf in df.groupby(["block_i", "block_j"]):
            color = sns.color_palette("hls", df.groupby(["block_i", "block_j"]).ngroups)[k]
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
    if not df.empty:
        for _, gdf in df.groupby(["block_i", "block_j"]):
            for i, row in gdf.iterrows():
                color = sns.color_palette("hls", df.groupby(["block_i", "block_j"]).ngroups)[k]
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
                                        row["ir"] * 2, row["jr"] * 2,
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
    txt = "Frame {} of {}".format(frmid, str(total_frames))
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
    if os.path.isdir(inp):
        frames = natsorted(glob(inp + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))
    regex = args.regex[0]

    # load roi and verify if its a file
    if args.roi[0]:
        is_roi_file = os.path.isfile(args.roi[0])

    # create the output path, if not present
    temp_path = os.path.abspath(args.temp_path[0])
    os.makedirs(temp_path, exist_ok=True)

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
                averages = averages[0:mframes]
                roi = roi.iloc[0:mframes]
            else:
                pass
        else:
            roi = False
            raise ValueError("Could not process region-of-interest file.")
    else:
        roi = False

    # select from which frame to start processing
    start = int(args.start[0])

    if int(args.nframes[0]) == -1:
        N = len(frames)
    else:
        N = int(args.nframes[0])
    total_frames = len(frames)

    frames = frames[start:start+N]
    averages = averages[start:start+N]

    # ----
    # full call for the detection function is

    # detector(frame, average, roi, output, cluster_pars=["dbscan", 10, 10],
    #              threshold_pars=["otsu"],
    #              total_frames=False, fit_kind="circle",
    #              block_shape=(256, 256), nthreads=1, debug=False, regex="[0-9]{6,}")
    # ----

    # debug - or serial case
    if args.debug:
        print("\n  + Detection in debug/serial mode")
        # ---- detect in serial mode ----
        frame_counter = 0
        for frame, average in zip(frames, averages):
            detector(frame, average, roi, temp_path,
                     cluster_pars, threshold_pars,
                     total_frames, FIT_KIND,
                     BLOCK_SHAPE, NTHREADS, regex=regex, debug=True)
            frame_counter += 1

    else:
        print("\n  + Detection in pararell mode")

        # plot in parallel mode
        if args.force_plot:
            debug = True
        else:
            debug = False

        # call
        fargs = zip(frames, averages,
                    repeat(roi), repeat(temp_path),
                    repeat(cluster_pars), repeat(threshold_pars),
                    repeat(total_frames), repeat(FIT_KIND),
                    repeat(BLOCK_SHAPE), repeat(NTHREADS), repeat(regex),
                    repeat(debug))
        with ProcessPool(max_workers=int(args.nproc[0]), max_tasks=99) as pool:
            for a in fargs:
                future = pool.schedule(detector, args=a,
                                       timeout=TIMEOUT)
                future.add_done_callback(task_done)

    # merge all cds files
    print("\n  + Merging outputs")
    dfs = []
    for fname in natsorted(glob(temp_path + "/*.csv")):
        dfs.append(pd.read_csv(fname))
    df = pd.concat(dfs)
    df.to_csv(args.output[0], index=False, chunksize=2**16)

    if not (args.debug or args.force_plot):
        shutil.rmtree(temp_path)


if __name__ == "__main__":

    print("\nDetecting wave breaking, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i", "--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with extracted frames.",)

    parser.add_argument("--regex", "-re", "-regex",
                        nargs=1,
                        action="store",
                        dest="regex",
                        required=False,
                        default=["[0-9]{6,}"],
                        help="Regex to search for frames. Default is [0-9]{6,}.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["wave_breaking_candidates.csv"],
                        required=False,
                        help="Output file name (csv).",)

    parser.add_argument("--subtract-averages", "--averages", "-averages",
                        nargs=1,
                        action="store",
                        dest="subtract_avg",
                        default=[False],
                        required=True,
                        help="Subtract average images from each frame."
                             "Must be path-like.",)

    parser.add_argument("--nproc", "-nproc",
                        nargs=1,
                        action="store",
                        dest="nproc",
                        default=[1],
                        required=False,
                        help="Number of processes to use.",)

    parser.add_argument("--nthreads", "-nthreads",
                        nargs=1,
                        action="store",
                        dest="nthreads",
                        default=[4],
                        required=False,
                        help="Number of threads to use.",)

    parser.add_argument("--cluster-method", "-cluster-method", "--cluster",
                        nargs="*",
                        action="store",
                        dest="cluster_kind",
                        default=["DBSCAN", 10, 10],
                        required=False,
                        help="Either DBSCAN or OPTICS. Default is DBSCAN. "
                             "Must also inform EPS and MIN_SAMPLES values.",)

    parser.add_argument("--threshold-method", "-threshold-method", "-trx", "--trx", "--threshold",
                        nargs="*",
                        action="store",
                        dest="threshold_method",
                        default=["adaptative", 11, 10],
                        required=False,
                        help="Thresholding method.",)

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
                        default=[128, 128],
                        required=False,
                        help="Size of the block to divide the image into."
                             " Must be a power of two. If not, will be cast.",)

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

    parser.add_argument("--force-plot-in-parallel-mode",
                        action="store_true",
                        dest="force_plot",
                        required=False,
                        help="If parsed, will run in debug mode.",)

    parser.add_argument("--frames-to-process", "-nframes", "--nframes",
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

    parser.add_argument("--timeout",
                        nargs=1,
                        action="store",
                        dest="timeout",
                        default=[3600],
                        required=False,
                        help="Kill a process if taking more than 3600 secs",)

    args = parser.parse_args()

    # Constants and options

    # which geometry to fit to wave breaking event candidates
    FIT_KIND = args.fitting_kind[0]

    # TODO: Re-factor this code latter

    # deal with thresholding method
    MTD = args.threshold_method[0]
    if MTD.lower() == "otsu":
        threshold_pars = ["otsu"]

    elif MTD.lower() == "entropy":
        threshold_pars = ["entropy"]

    elif MTD.lower() == "adaptative":
        if len(args.threshold_method) < 3:
            raise ValueError("Must give window size and offset values.")
        threshold_pars = ["adaptative", int(args.threshold_method[1]),
                          int(args.threshold_method[2])]

    elif MTD.lower() == "constant":
        if len(args.threshold_method) < 2:
            raise ValueError("Must give the constant value.")
        threshold_pars = ["constant", int(args.threshold_method[1])]

    elif MTD.lower() == "file":
        if len(args.threshold_method) < 2:
            raise ValueError("Must give the file path.")
        threshold_pars = ["file", args.threshold_method[1]]

    else:
        raise ValueError("Threshold method is unknown.")

    print("  - Thresholding method is: {}".format(MTD))
    if args.threshold_method[1:]:
        print("  - Thresholding parameters are: ", args.threshold_method[1:])

    # deal with clustering method
    CLUSTER_KIND = args.cluster_kind[0]
    if CLUSTER_KIND.lower() == "dbscan":
        if len(args.cluster_kind) < 3:
            raise ValueError("Must give eps and min_samples values.")
        eps = float(args.cluster_kind[1])
        min_samples = int(args.cluster_kind[2])
        cluster_pars = ["dbscan", eps, min_samples]

    elif CLUSTER_KIND.lower() == "hdbscan":
        if len(args.cluster_kind) < 2:
            raise ValueError("Must give min_samples value.")
        min_samples = int(args.cluster_kind[1])
        cluster_pars = ["hdbscan", False, min_samples]

    elif CLUSTER_KIND.lower() == "optics":
        if len(args.cluster_kind) < 2:
            raise ValueError("Must give min_samples value.")
        eps = float(args.cluster_kind[1])
        min_samples = int(args.cluster_kind[2])
        cluster_pars = ["optics", eps, min_samples]

    else:
        raise NotImplementedError("Only DBSCAN/HDBSCAN/OPTICS are currently working.")

    print("\n  - Clustering method is: {}".format(CLUSTER_KIND.lower()))
    if args.cluster_kind[1:]:
        print("  - Clustering parameters are: ", args.cluster_kind[1:])

    # image blovk size to process the image in chunks
    # this avois memory errors
    BLOCK_SHAPE = [int(args.block_shape[0]), int(args.block_shape[0])]

    # maximum time allowed for each function call
    TIMEOUT = int(args.timeout[0])

    # global number of threads
    NTHREADS = int(args.nthreads[0])

    # this locks sklearn and makes sure it does not create more processes than
    # what it"s being asked for.
    parallel_backend("threading", n_jobs=NTHREADS)

    # call the main program
    main()

    print("\nMy work is done!\n")
