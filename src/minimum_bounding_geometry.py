# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : minimum_bounding_geometry.py
# POURPOSE : compute the minimum bounding geometry of surface file for each
#            timestep. Use a ConvexHull approach
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# V1.0     : XX/XX/XXXX [Caio Stringari]
#
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# uncomment these lines to run on the grid
# import matplotlib as mpl
# mpl.use("Agg")
import os
import ast

import argparse

import numpy as np

import xarray as xr
import pandas as pd

from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def call_main_with_surfaces_file():
    """Call the main program."""
    # open the dataset
    ds = xr.open_dataset(args.input[0])

    # open a reference frame, for debuging only
    if args.debug:
        frame = plt.imread(args.frame[0])
        outpath = "debug_mbg"
        os.makedirs(outpath, exist_ok=True)

    # scale
    scale = float(args.scale[0])

    # timeloop
    top_left_i = []
    top_left_j = []
    length = []
    width = []
    for t, time in enumerate(ds["T"].values):

        print("  - processing time {} of {}".format(t + 1, len(ds["T"].values)),
              end="\r")

        # slice in time
        tds = ds.isel(T=t)

        # load variables
        xgrid = tds["iR"].values
        ygrid = tds["jR"].values
        z = tds["Z"].values

        # mask
        z[z == z.min()] = np.nan

        # open a debug plot
        if args.debug:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.pcolormesh(tds["X_grid"], tds["Y_grid"], tds["Z"],
                           vmin=-10, vmax=10, cmap="RdBu_r")
            ax2.imshow(frame, cmap="Greys_r")

            ax1.set_aspect("equal")

        try:
            # compute a convex hull of the grid
            points = np.vstack([xgrid[~np.isnan(z)].flatten(),
                                ygrid[~np.isnan(z)].flatten()]).T
            hull = ConvexHull(points)
            hull.close()
            # hull vertices
            vertices = np.vstack([points[hull.vertices, 0],
                                  points[hull.vertices, 1]]).T
            vertices = np.insert(vertices, -1,
                                 [vertices[0, 0], vertices[0, 1]],
                                 axis=0)

            # get a rectangle
            imin = vertices[:, 0].min()
            jmin = vertices[:, 1].min()
            dx = vertices[:, 0].max() - vertices[:, 0].min()
            dy = vertices[:, 1].max() - vertices[:, 1].min()

            # scale
            dxnew = dx * scale
            dynew = dy * scale
            inew = imin + dx / 2 - dxnew / 2
            jnew = jmin + dy / 2 - dynew / 2

            # append to output
            top_left_i.append(int(np.ceil(inew)))
            top_left_j.append(int(np.ceil(jnew)))
            length.append(int(dxnew))
            width.append(int(dynew))

            # add to plot
            if args.debug:
                rec_patch = patches.Rectangle((inew, jnew), dxnew, dynew,
                                              linewidth=2,
                                              edgecolor='r',
                                              facecolor='none',
                                              linestyle="--")
                ax2.plot(vertices[:, 0], vertices[:, 1], 'r-', lw=3)
                ax2.add_patch(rec_patch)

                plt.savefig("{}/{}".format(outpath, str(t).zfill(6)))
                plt.close()

        except Exception:

            print("    - error in frame {} \n".format(t))
            top_left_i.append(np.nan)
            top_left_j.append(np.nan)
            length.append(np.nan)
            width.append(np.nan)

    # save output file
    df = pd.DataFrame(np.vstack([top_left_i, top_left_j, length, width]).T,
                      columns=["i", "j", "width", "height"])
    df.index.name = "frame"
    df.to_csv(args.output[0])


if __name__ == "__main__":

    print("\nExtracting minimum geometry, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input netcdf file
    parser.add_argument("--use-surface-file",
                        nargs=1,
                        action="store",
                        dest="has_surface",
                        default=[True],
                        required=False,
                        help="Use Surface file to extract MBG?.",)

    # input netcdf file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=False,
                        default=[False],
                        help="Input netcdf file.",)

    parser.add_argument("--frame", "-f",
                        nargs=1,
                        action="store",
                        dest="frame",
                        required=False,
                        help="Input reference frame for debuging.",)

    parser.add_argument("--debug",
                        action="store_true",
                        dest="debug",
                        help="Will show plots if true.",)

    parser.add_argument("--scale", "-scale",
                        nargs=1,
                        action="store",
                        dest="scale",
                        default=[1],
                        help="A scale factor to shrink the ROI.",)

    parser.add_argument("--user-define-coordinates",
                        nargs=4,
                        action="store",
                        default=[0, 0, 0, 0],
                        dest="user_coords",
                        help="User defined MBG."
                             "top_left_i, top_left_j, length, width",)

    parser.add_argument("--repeat",
                        nargs=1,
                        action="store",
                        dest="repeat",
                        default=[18000],
                        help="How many lines to repeat the MGB.")

    # output file
    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output csv file.",)

    args = parser.parse_args()

    has_surface = ast.literal_eval(args.has_surface[0])

    if has_surface:
        call_main_with_surfaces_file()
    else:

        # open a reference frame, for debuging only
        if args.debug:
            frame = plt.imread(args.frame[0])

        # scale
        scale = float(args.scale[0])

        coords = np.array(args.user_coords).astype(int)

        # get a rectangle
        imin = coords[0]
        jmin = coords[1]
        dx = coords[2]
        dy = coords[3]

        # scale
        dxnew = dx * scale
        dynew = dy * scale
        inew = imin + dx / 2 - dxnew / 2
        jnew = jmin + dy / 2 - dynew / 2

        # organize
        top_left_i = imin
        top_left_j = jmin
        length = dxnew
        width = dynew

        df = pd.DataFrame(columns=["i", "j", "width", "height"])
        df["i"] = [top_left_i] * int(args.repeat[0])
        df["j"] = [top_left_j] * int(args.repeat[0])
        df["width"] = [length] * int(args.repeat[0])
        df["height"] = [width] * int(args.repeat[0])
        df.index.name = "frame"
        df.to_csv(args.output[0])

        # plot
        if args.debug:
            rec_patch = patches.Rectangle((inew, jnew), dxnew, dynew,
                                          linewidth=2,
                                          edgecolor='r',
                                          facecolor='none',
                                          linestyle="--")
            fig, ax = plt.subplots()
            ax.imshow(frame, cmap="Greys_r")
            ax.add_patch(rec_patch)
            plt.show()

    print("\n\nMy work is done!\n")
