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

    # input configuration file
    parser.add_argument("--model-results", "-res", "--results",
                        nargs=1,
                        action="store",
                        dest="results",
                        required=True,
                        help="Model results.",)

    # input model
    parser.add_argument("--history", "-hist",
                        nargs=1,
                        action="store",
                        dest="hist",
                        required=True,
                        help="Model history.",)

    parser.add_argument("--overfit", "-overfit",
                        nargs=1,
                        action="store",
                        dest="overfit",
                        required=False,
                        default=[100],
                        help="Model overfit epoch.",)

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

    overfit = int(args.overfit[0])

    res = pd.read_csv(args.results[0])

    print(classification_report(res["true"], res["prediction"]))

    # compute the confusion matrix
    cm = confusion_matrix(res["true"], res["prediction"],
                          normalize="pred", labels=[1, 0],)
    df_cm = pd.DataFrame(
        cm, index=[i for i in ["Active Wave Breaking", "Otherwise"]],
        columns=[i for i in ["Active Wave Breaking", "Otherwise"]])

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

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

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

    # CM
    sns.heatmap(df_cm, annot=True, cmap="Blues", linewidths=1, linecolor="k",
                cbar_kws={"pad": 0.015}, square=False, ax=ax3, fmt="0.2f")
    ax3.set_yticklabels(ax3.get_yticklabels(), va="center", fontsize=10,
                        rotation=90)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0, fontsize=10)

    ax3.set_xlabel("Predicted label")
    ax3.set_ylabel("True label")

    # finalize
    for ax in [ax1, ax2]:
        sns.despine(ax=ax)
    k = 0
    letters = ["a)", "b)", "c)"]
    for ax in [ax1, ax2, ax3]:
        ax.text(0.95, 0.95, letters[k],
                va="top", zorder=100, transform=ax.transAxes, ha="right",
                bbox=dict(
                    boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))
        k += 1
    # sns.despine(ax=ax3, top=True, bottom=True, left=True, right=True)
    # ax3.grid(False)

    fig.tight_layout()
    plt.savefig(args.output[0],  bbox_inches="tight", pad=0.1, dpi=300)
    plt.show()

    print("\nMy work is done!\n")
