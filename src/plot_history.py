"""
Plot model history and confusion matrix

# PROGRAM   : test_wave_breaking_classifier.py
# POURPOSE  : classify wave breaking using a convnets
# AUTHOR    : Caio Eadi Stringari
# EMAIL     : caio.stringari@gmail.com
# V1.0      : 05/05/2020 [Caio Stringari]
"""
import argparse

import pandas as pd

import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {"axes.linewidth": 2,
                        "legend.frameon": True,
                        "axes.facecolor": "#E9E9F1",
                        "grid.color": "w"})
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams['patch.edgecolor'] = "k"
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["dodgerblue", "orangered",
                                                    "k"])


def smooth(a, n):
    """Smooth a curve rolling average."""
    return pd.Series(a).rolling(window=n, center=True).mean().to_numpy()


if __name__ == '__main__':

    print("\nClassifiying wave breaking data, please wait...\n")

    # Argument parser
    parser = argparse.ArgumentParser()

    # history
    parser.add_argument("--history", "-hist",
                        nargs=1,
                        action="store",
                        dest="hist",
                        required=True,
                        help="Model history.",)

    parser.add_argument("--smooth", "-smooth",
                        nargs=1,
                        action="store",
                        dest="smooth",
                        required=False,
                        default=[10],
                        help="Number of samples to smooth the curve.",)
    # output model
    parser.add_argument("--output", "-o",
                        nargs=1,
                        action="store",
                        dest="output",
                        required=True,
                        help="Output figure name.",)

    args = parser.parse_args()

    SMTH = int(args.smooth[0])

    hist = pd.read_csv(args.hist[0])

    # get history
    epochs = hist["epoch"]
    train_loss = hist["loss"]
    train_loss_smth = smooth(train_loss, SMTH)
    val_loss = hist["val_loss"]
    val_loss_smth = smooth(val_loss, SMTH)

    train_auc = hist["AUC"]
    train_auc_smth = smooth(train_auc, SMTH)
    val_auc = hist["val_AUC"]
    val_auc_smth = smooth(val_auc, SMTH)

    overfit = np.argmin(val_loss)  # int(args.overfit[0])

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_loss, alpha=0.5, color="indigo", lw=2)
    ax1.plot(epochs, train_loss_smth, lw=3, color="indigo", label="Train")
    ax1.plot(epochs, val_loss, alpha=0.5, color="forestgreen", lw=2)
    ax1.plot(epochs, val_loss_smth, lw=3, color="forestgreen",
             label="Validation")
    ax1.axvline(overfit, lw=3, color="k", ls="--", label="Overfitting")
    ax1.set_xlabel("Epoch $[-]$")
    ax1.set_ylabel("Loss value $[-]$")
    lg = ax1.legend(loc=2, fontsize=12)
    lg.get_frame().set_color("w")
    ax1.grid(color="w", ls="-", lw=2)
    ax1.set_ylim(top=1)

    # AUC
    ax2.plot(epochs, train_auc, alpha=0.5, color="indigo", lw=2)
    ax2.plot(epochs, train_auc_smth, lw=3, color="indigo", label="Train")
    ax2.plot(epochs, val_auc, alpha=0.5, color="forestgreen", lw=2)
    ax2.plot(epochs, val_auc_smth, lw=3, color="forestgreen",
             label="Validation")
    ax2.axvline(overfit, lw=3, color="k", ls="--", label="Overfitting")
    ax2.set_xlabel("Epoch $[-]$")
    ax2.set_ylabel("AUC $[-]$")
    lg = ax2.legend(loc=3, fontsize=12)
    lg.get_frame().set_color("w")
    ax2.grid(color="w", ls="-", lw=2)
    ax2.set_ylim(0.5, 1)

    for ax in [ax1, ax2]:
        ax.text(0.95, 0.95, "Overfitting starts at epoch {}".format(overfit),
                va="top", zorder=100, transform=ax.transAxes, ha="right",
                bbox=dict(
                    boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))

    # finalize
    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
    fig.tight_layout()
    plt.savefig(args.output[0],  bbox_inches="tight", pad=0.1, dpi=300)
    plt.show()

    print("\nMy work is done!\n")
