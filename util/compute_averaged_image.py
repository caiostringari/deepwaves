# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
#
# SCRIPT   : compute_averaged image.py
# POURPOSE : Compute image average
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
from glob import glob
from natsort import natsorted

import numpy as np

from PIL import Image

import multiprocessing


def average_worker(imlist, output_file_name):
    """Average a sequence of images using numpy and PIL."""
    images = np.array([np.array(Image.open(fname)) for fname in imlist])
    arr = np.array(np.mean(images, axis=(0)), dtype=np.uint8)
    out = Image.fromarray(arr)
    out.save(output_file_name)


def main():
    """Call the main program."""
    # verify if the input path exists,
    # if it does, then get the frame names
    inp = args.input[0]
    if os.path.isdir(inp):
        frames = natsorted(glob(inp + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(inp))

    # create the output path, if not present
    outpath = os.path.abspath(args.output[0])
    os.makedirs(outpath, exist_ok=True)

    # get number of frames to use for averaging
    nframes = np.int(args.nframes[0])

    # get number of cores to use
    nproc = np.int(args.nproc[0])

    # split the list of input frames into N lists with nframes per list
    lenght = np.int(np.floor(len(frames) / nframes))
    frame_chunks = np.array_split(frames, lenght)

    # check
    nframes_check = 0
    for chunk in frame_chunks:
        nframes_check += len(chunk)
    if nframes_check != len(frames):
        raise ValueError("Problem with frame segmentation.")

    # create output names
    avg_out_names = []
    for i, chunk in enumerate(frame_chunks):
        name = os.path.join(outpath, "avg_{}.png".format(str(i).zfill(4)))
        avg_out_names.append(name)

    # create a process pool
    p = multiprocessing.Pool(nproc)

    # compute average
    p.starmap(average_worker, zip(frame_chunks,  avg_out_names))


if __name__ == "__main__":

    print("\nComputing average, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument("--input", "-i",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input folder with images file.",)

    # input frame file
    parser.add_argument("--nframes", "-n",
                        nargs=1,
                        action="store",
                        dest="nframes",
                        required=True,
                        help="Number of frames to use for the average.",)

    parser.add_argument("--nproc", "-nproc",
                        nargs=1,
                        action="store",
                        dest="nproc",
                        default=[False],
                        required=False,
                        help="Number of processes to use.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        default=["avg/"],
                        required=False,
                        help="Output path for images.",)

    args = parser.parse_args()

    main()
