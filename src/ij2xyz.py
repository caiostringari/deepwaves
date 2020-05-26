# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : ij2xcyz.py
# POURPOSE : convert pixel coordinates to metric coordinates
# AUTHOR   : Caio Eadi Stringari
#
# V1.0     : XX/XX/XXXX [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
import argparse

import os

import numpy as np

import pandas as pd

import xarray as xr

import scipy.spatial

import matplotlib.pyplot as plt


def rotation_transform(theta):
    """
    Build a 2-D rotation matrix transformation given an input angle.

    Parameters:
    ----------
    theta : float
        rotation angle in degrees.

    Returns:
    -------
    R : array
        rotation matrix
    """
    theta = np.radians(theta)
    R = [[np.math.cos(theta), -np.math.sin(theta)],
         [np.math.sin(theta), np.math.cos(theta)]]

    return np.array(R)


def map_ij_to_xy(igrid, jgrid, xgrid,  ygrid, i, j, verbose=1):
    """
    Map (i,j) to (x,y) coordinates.

    Parameters:
    ----------
    igrid, jgrid : np.array (2D)
        i,j coordinates of the grid.

    xgrid, ygrid : np.array (2D)
        x,y coordinates of the grid.

    i, j: float or int
        pixel coordinates

    Returns:
    -------
    x, y : np.array
        metric coordinates
    """
    # remove nans
    igrid[np.isnan(igrid)] = -999
    jgrid[np.isnan(igrid)] = -999
    igrid[np.isnan(jgrid)] = -999
    jgrid[np.isnan(jgrid)] = -999

    Tree = scipy.spatial.KDTree(np.vstack([igrid.flatten(),
                                           jgrid.flatten()]).T)

    # search for nearest neighbour
    distance, index = Tree.query([i, j])

    i_idx = np.unravel_index(index, igrid.shape)[0]
    j_idx = np.unravel_index(index, jgrid.shape)[1]

    if verbose == 1:
        print("     target: i={} j={}".format(i, j))
        print("     nearest neighbour: i={} j={}".format(
            igrid[i_idx, j_idx], jgrid[i_idx, j_idx]))
        print("     nearest neighbour: x={} y={}".format(
            xgrid[i_idx, j_idx], ygrid[i_idx, j_idx]))

    x = xgrid[i_idx, j_idx]
    y = ygrid[i_idx, j_idx]

    return x, y


if __name__ == "__main__":

    print("\nConverting coordinates, please wait...")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input netcdf file.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output csv file.",)

    # input coordinates
    parser.add_argument("--coordinates", "-coords", "-point", "---points",
                        nargs="*",
                        action="store",
                        dest="coordinates",
                        required=True,
                        help="Input coordinates."
                             "Either a csv file or a i,j pair.",)

    parser.add_argument("--rotate", "-r",
                        nargs=1,
                        action="store",
                        dest="rotation",
                        default=[False],
                        required=False,
                        help="Grid rotation angle. Default is false.",)

    args = parser.parse_args()

    # check if input is file
    isfile = False
    if os.path.isfile(args.coordinates[0]):
        isfile = True

    # open the dataset
    ds = xr.open_dataset(args.input[0])

    # read the grid
    try:
        xgrid = ds["X_grid"].values
        ygrid = ds["Y_grid"].values
    except Exception:
        raise ValueError("Netcdf coordinate names not supported.")

    # rotate the grid, if asked to
    if args.rotation[0]:
        print("\n  - Rotating grid by {} degrees".format(args.rotation[0]))

        # rotate
        R = rotation_transform(float(args.rotation[0]))
        XY = np.dot(R, np.vstack([xgrid.flatten(),
                                  ygrid.flatten()])).T

        # reshape
        xshape = xgrid.shape
        yshape = ygrid.shape
        xgrid = XY[:, 0].reshape(xshape)
        ygrid = XY[:, 1].reshape(yshape)

    if isfile:
        df = pd.read_csv(args.coordinates[0])

        targets = ["ic", "jc", "frame"]
        for t in targets:
            if t not in df.keys():
                raise ValueError(
                    "Input data must have a column named \'{}\'".format(t))

        k = 0
        X = []
        Y = []
        print("\n  - Looping over Dataframe rows.")
        # loop over dataframe coordinates
        for t, i, j in zip(df["frame"].values,
                           df["ic"].values,
                           df["jc"].values):

            print("\n   - Step {} of {}".format(k+1, len(df["frame"].values)))

            # extract grid coordinates
            igrid = ds["iR"].isel(T=t).values
            jgrid = ds["jR"].isel(T=t).values

            x, y = map_ij_to_xy(igrid, jgrid, xgrid,  ygrid, i, j)

            X.append(x)
            Y.append(y)

            k += 1

        # update dateframe
        df["xc"] = X
        df["yc"] = Y
        df.to_csv(args.output[0], index=False)

    # requested coordinates
    else:
        i = int(args.coordinates[0])
        j = int(args.coordinates[1])

        # time loop
        print("\n  - Timeloop")
        F = []
        X = []
        Y = []
        for t, time in enumerate(ds["T"].values):
            print("\n   - Step {} of {}".format(t+1, len(ds["T"].values)))

            # extract grid coordinates
            igrid = ds["iR"].isel(T=t).values
            jgrid = ds["jR"].isel(T=t).values

            x, y = map_ij_to_xy(igrid, jgrid, xgrid,  ygrid, i, j)

            X.append(x)
            Y.append(y)
            F.append(t)

        df = pd.Dataframe(np.vstack([F, X, Y]).T, columns=["frame", "x", "y"])
        df.to_csv(args.output[0], index=False)

    # # confirm results
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax1.pcolormesh(xgrid, ygrid, ds.isel(T=0)["Z"])
    # ax1.scatter(X[0], Y[0], 100, color="k", marker="+", linewidths=4)
    #
    # ax2.pcolormesh(igrid, jgrid, ds.isel(T=0)["Z"])
    # ax2.scatter(i, j, 100, color="k", marker="+", linewidths=4)
    #
    # plt.show()



    print("\n\nMy work is done!\n")
