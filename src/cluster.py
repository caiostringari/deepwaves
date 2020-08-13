r"""
Cluster wave breaking events in time and space. This script can use the results
of naive_wave_breaking_detector directly but this is not recommended.

It is recommended that you narrow down the candidates for clustering
using predict_active_wave_breaking or another tool.

For help: python cluster_wave_breaking_events.py --help

Usage:
python cluster_wave_breaking_events.py -i "robust_wave_breaking_events.csv"   \
                                       -o "clustered_wave_breaing_events.csv" \
                                       --cluster method "DBSCAN"              \
                                       --eps 10 -min-samples 10

-i : input csv file from predict_active_wave_braeking_v2

-o : output file name

--cluster-method : Only DBSCAN is implemented currently

--eps : eps parameter for DBSCAN or OPTICS

--min-samples : min_samples parameter for DBSCAN or OPTICS

--chunk-size : Maximum nuber of rows to read from the file at a given time

The output CSV columns are organized are the same as described in
"naive_wave_breaking_detector"

The only addition is:
    - wave_breaking_event : unique wave breaking event (space and time).

# SCRIPT   : cluster_wave_breaking.py
# POURPOSE : cluster the results of wave breaking detection
# AUTHOR   : Caio Eadi Stringari
# V2.0     : 15/04/2020 [Caio Stringari]
"""
import argparse
import numpy as np

# ML
from sklearn.cluster import DBSCAN, OPTICS

# pandas for I/O
import pandas as pd

# used only for debug
import matplotlib.patches as patches


def compute_patch(row):
    """
    Compute points inside an ellipse (or circle)

    Parameters:
    ----------
    row : pd.Series
        series with data

    Returns:
    -------
    ic, jc: np.array
        arrays with center points of the ellipse
    points : np.ndarray
        coordinates of points inside the ellipse
    """
    # check if circle or ellipse
    try:
        r1 = row["ir"]
        r2 = row["jr"]
    except Exception:
        r1 = row["ir"]
        r2 = r1

    # circle
    if r1 == r2:
        c = patches.Circle((row["ic"], row["jc"]), r1)
    # ellipse case
    else:
        c = patches.Ellipse((row["ic"], row["jc"]), row["ir"]*2, row["jr"]*2,
                            angle=row["theta_ij"])

    # populate the ellipse with points
    rmax = max(r1, r2)
    x = np.arange(row["ic"]-rmax, row["ic"]+rmax+1, 1)
    y = np.arange(row["jc"]-rmax, row["jc"]+rmax+1, 1)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.flatten(), Y.flatten()]).T

    ins = c.contains_points(points)
    points = points[ins]

    return row["ic"], row["jc"], points


def main():
    """Call the main program."""

    df = pd.read_csv(INPUT)

    # select only active wave breaking
    df = df.loc[df["class"] == 1]

    # ckeck if all needed keys are present
    targets = ["ic", "jc", "ir", "frame"]
    for t in targets:
        if t not in df.keys():
            raise ValueError(
                "Key \'{}\' must be present in the data.".format(t))

    # split the dataframe into chunks of CHUNK size
    dfs = [df[i:i+CHUNK] for i in range(0, df.shape[0], CHUNK)]

    # loop over dataframes
    k = 0
    events = []  # spatio-temporal clusters
    for kdf in dfs:
        print("  - Processing chunk {} of {}".format(k+1, len(dfs)), end="\r")

        # compute pathches
        groups = kdf.groupby("frame")

        Iclf = []
        Jclf = []
        Ic = []
        Jc = []
        Tclf = []

        # loop over timesteps
        for g, gdf in groups:
            # loop over ellipses
            for r, row in gdf.iterrows():
                try:
                    ic, jc, points = compute_patch(row)
                    # append for clustering
                    for point in points:
                        Iclf.append(point[0])
                        Jclf.append(point[1])
                        Tclf.append(g)
                        Ic.append(ic)
                        Jc.append(jc)
                except Exception:
                    pass

        # cluster
        X = pd.DataFrame(np.vstack([Iclf, Jclf, Ic, Jc, Tclf]).T,
                         columns=["i", "j", "ic", "jc", "frame"])
        clf = OPTICS(cluster_method="dbscan",
                     metric="euclidean",
                     eps=EPS,
                     max_eps=EPS,
                     min_samples=MIN_SAMPLES,
                     min_cluster_size=MIN_SAMPLES,
                     n_jobs=NJOBS,
                     algorithm="ball_tree").fit(X)
        # clf = DBSCAN(eps=EPS,
        #              metric="euclidean",
        #              min_samples=MIN_SAMPLES,
        #              n_jobs=NJOBS,
        #              algorithm="ball_tree").fit(X[["i", "j", "frame"]])
        X["wave_breaking_event"] = clf.labels_

        # reorganize data
        key = ["ic", "jc", "frame", "wave_breaking_event"]
        for event in X[key].drop_duplicates()["wave_breaking_event"].values:
            events.append(event)
        k += 1
        # break

    df = dfs[0]
    df["wave_breaking_event"] = events
    df.to_csv(OUTPUT, index=False)


if __name__ == "__main__":

    print("\nClustering wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input CSV file with detected wave breaking.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["clusters.csv"],
                        required=False,
                        help="Output file name.",)

    parser.add_argument("--njobs", "-njobs",
                        nargs=1,
                        action="store",
                        dest="njobs",
                        default=[-1],
                        required=False,
                        help="Number of jobs for DBSCAN.",)

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
                        default=[20],
                        required=False,
                        help="DBSCAN min_samples parameter (pixels)",)

    parser.add_argument("--chunk-size", "-chunk-size",
                        nargs=1,
                        action="store",
                        dest="chunk",
                        default=[1000],
                        required=False,
                        help="Chunk size to process at a time.",)

    args = parser.parse_args()

    # Constants
    INPUT = args.input[0]
    CHUNK = int(args.chunk[0])
    EPS = float(args.eps[0])
    MIN_SAMPLES = int(args.min_samples[0])
    NJOBS = int(args.njobs[0])
    OUTPUT = args.output[0]

    if args.cluster_kind[0].lower() != "DBSCAN".lower():
        raise NotImplementedError(
            "{} is not implemented yet.".format(args.cluster_kind[0]))

    main()

    print("\n\nMy work is done!\n")
