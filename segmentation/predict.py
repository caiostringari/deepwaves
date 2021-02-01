"""
Use a pre-trained segmentation model. Make sure your input is 256x256.

PROGRAM   : predict.py
POURPOSE  : Get the regions in an image where waves are actively breaking
AUTHOR    : Caio Eadi Stringari
EMAIL     : caio.stringari@gmail.com
V2.0      : 06/10/2020 [Caio Stringari]
"""

import os
import argparse

from glob import glob
from natsort import natsorted

import numpy as np

from skimage.io import imread
from skimage.color import grey2rgb

import tensorflow as tf

# progress bar
from tqdm import tqdm

# quite skimage warnings
import warnings

# plot
import matplotlib.pyplot as plt

tf.get_logger().setLevel('INFO')
warnings.filterwarnings("ignore")


def display_mask(val_preds, i):
    """Display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return mask


def main():
    """Call the main program."""
    # i/o
    model = args.model[0]  # pre-trained model
    inp_data = args.input[0]  # frames to be segmented
    out_data = args.output[0]  # output csv file

    # create output
    os.makedirs(out_data, exist_ok=True)

    # load the model
    M = tf.keras.models.load_model(model)

    # verify if the input path exists,
    # if it does, then get the frame names
    if os.path.isdir(inp_data):
        images = natsorted(glob(inp_data + "/*"))
    else:
        raise IOError("No such file or directory \"{}\"".format(inp_data))

    # --- loop over frames ---
    pbar = tqdm(total=len(images))

    for k, image in enumerate(images):

        # print("-- plotting frame {} of {}".format(k+1, total_frames), end="\r")

        # load image
        img = grey2rgb(imread(image))

        # predict
        pred = M.predict(np.expand_dims(img/255, axis=0))  # very important to normalize your data !
        prd = np.squeeze(np.argmax(pred, axis=-1))

        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                       sharex=True, sharey=True)
        ax1.imshow(np.squeeze(img))
        ax2.imshow(np.squeeze(prd))
        fig.tight_layout()
        plt.savefig(os.path.join(out_data, str(k).zfill(6) + ".png"),
                    pad_inches=0.1, bbox_inches='tight')
        plt.close()

        pbar.update()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict active wave breaking segmentation')

    parser.add_argument('--model', "-M",
                        nargs=1,
                        dest='model',
                        help='pre-trained model in .h5 format',
                        required=True,
                        action='store')

    parser.add_argument("--input", "-i", "--frames", "-frames",
                        nargs=1,
                        action="store",
                        dest="input",
                        required=True,
                        help="Input path with data.",)

    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output path.",)

    args = parser.parse_args()

    main()
