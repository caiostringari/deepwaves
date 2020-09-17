"""
Interpret the results of the classifier using GradCAM.

You will need to download a pre-trained model.

# PROGRAM   : interpret.py
# AUTHOR    : Caio Eadi Stringari
# EMAIL     : caio.stringari@gmail.com
# V1.0      : 10/08/2020 [Caio Stringari]
"""
import os

import argparse

import numpy as np

import tensorflow as tf
from tensorflow import keras

import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


class GradCAM:
    """
    Implements GradCAM.

    reference: https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    """

    def __init__(self, model, layerName):
        """Initialize the model."""
        self.model = model
        self.layerName = layerName

        self.gradModel = keras.models.Model(inputs=[self.model.inputs],
                                            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

    def compute_heatmap(self, image, classIdx, eps=1e-8):
        """Compute a heatmap with the class activation."""
        with tf.GradientTape() as tape:
            tape.watch(self.gradModel.get_layer(self.layerName).output)
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = self.gradModel(inputs)

            if len(predictions) == 1:
                # binary Classification
                loss = predictions[0]
            else:
                loss = predictions[:, classIdx]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap


if __name__ == '__main__':

    print("\nClassifiying wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # input configuration file
    parser.add_argument("--model", "-M",
                        action="store",
                        dest="model",
                        required=True,
                        help="Input model in .h5 format.",)

    # input model
    parser.add_argument("--data", "-data",
                        action="store",
                        dest="data",
                        required=True,
                        help="Input path with image data.",)

    # output model
    parser.add_argument("--output", "-o",
                        action="store",
                        dest="output",
                        required=True,
                        help="Output path.",)

    args = parser.parse_args()

    # --- model ---
    model = tf.keras.models.load_model(args.model)

    datagen = ImageDataGenerator(rescale=1./255.)
    inp_shape = model.input_shape
    img_height = inp_shape[1]  # image height for all images
    img_width = inp_shape[2]  # image width for all images

    print("\n    Fitting the teset data generator:\n")
    generator = datagen.flow_from_directory(
        directory=args.data, batch_size=1, shuffle=False,
        target_size=(img_height, img_width),
        class_mode="binary")

    # Initialize CAM
    cam = GradCAM(model, "block5_conv3")

    # output
    out = args.output
    if not os.path.isdir(out):
        os.makedirs(out, exist_ok=True)

    k = 0
    for step in range(generator.n):

        # get the image
        X, y = generator.next()
        label = model.predict(X)

        # compute the heatmap
        heatmap = cam.compute_heatmap(X, y, eps=1e-20)

        # plot
        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(X), alpha=1)
        ax.imshow(heatmap, alpha=0.6, cmap="magma")
        plt.axis('off')
        plt.savefig(
            os.path.join(out, os.path.basename(generator.filenames[k])),
            dpi=300, pad_inches=0, bbox_inches='tight')

        k += 1

    print("\nMy work is done!\n")
