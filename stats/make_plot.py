import sys

import numpy as np

import statsmodels.api as sm
from scipy import stats

import pickle

from string import ascii_lowercase

# import numpy.polynomial.polynomial as poly

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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


def running_mean(x, N):
    """Compute numpy running mean."""
    out = np.zeros_like(x, dtype=np.float64)
    dim_len = x.shape[0]
    for i in range(dim_len):
        if N % 2 == 0:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 2
        else:
            a, b = i - (N - 1) // 2, i + (N - 1) // 2 + 1

        # cap indices to min and max indices
        a = max(0, a)
        b = min(dim_len, b)
        out[i] = np.mean(x[a:b])
    return out


if __name__ == '__main__':

    # -- load data --
    inp = sys.argv[1]
    out = sys.argv[2]
    with open(inp, 'rb') as f:
        df = pickle.load(f)

    # -- group Lambda by wind speed --
    target_winds = [6.06, 8.71, 15.25]
    df = df.loc[df['wnd'].isin(target_winds)]

    # colors
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=16)

    # -- compute Lambda --
    dcb = 0.1
    cb = np.arange(dcb, 10 + dcb, dcb)
    Lambda = []
    for wnd, wdf in df.groupby("wnd"):

        if wnd in target_winds:
            Atot = np.mean(wdf['svA'].values)
            Ttot = np.mean(wdf['svT'].values)

            crest_length = (wdf["Lb"].values *
                            wdf['DT'].values) / (Atot * Ttot * dcb)

            _Lambda = np.zeros(cb.shape)
            for i in range(0, len(cb)):
                fcb = np.argwhere((wdf["cb"].values >= cb[i]) & (
                    wdf["cb"].values < cb[i] + dcb))
                if len(fcb) > 1:
                    _Lambda[i] = np.nansum(crest_length[fcb[:, 0]])
            Lambda.append(_Lambda)

    # -- plot --
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
          ) = plt.subplots(2, 3, figsize=(11, 6))

    # duration
    xpdf = np.arange(0, 4, 0.01)
    pars = stats.weibull_min.fit(df["DT"])
    ypdf = stats.weibull_min.pdf(xpdf, *pars)
    kde = sm.nonparametric.KDEUnivariate(df["DT"])
    kde.fit(bw=0.05)
    mode = kde.support[np.argmax(kde.density)]
    label = r"Weibull PDF"

    ax1.hist(df["DT"], density=True, bins=np.arange(
        0, 2, 0.1), color="0.15", alpha=0.5)
    ax1.axvline(df["DT"].mean(), color="indigo", lw=3, ls="--",
                label="Mean={0:.1f}s".format(df["DT"].mean()))
    ax1.axvline(mode, color="coral", lw=3, ls="--",
                label="Mode={0:.1f}s".format(mode))
    ax1.plot(xpdf, ypdf, color="dodgerblue", lw=3, label=label)
    # ax1.plot(kde.support, kde.density, lw=3, color="k", ls="-", label="KDE")
    ax1.set_xlabel(r"$T_{br}$ $[s]$")
    ax1.set_ylabel(r"$p(T_{br})$ $[s]$")
    ax1.set_xlim(0, 2)
    lg = ax1.legend(loc=1, fontsize=10)
    lg.get_frame().set_color("w")

    # area
    pars = stats.pareto.fit(df["Ab"])
    pareto_x = np.arange(0.1, 20, 0.1)
    ypdf = stats.pareto.pdf(pareto_x, *pars)
    label = r"Pareto PDF"

    ax2.hist(df["Ab"], density=True, bins=np.arange(
        0, 10, 0.5), color="0.15", alpha=0.5)
    ax2.plot(pareto_x, ypdf, color="dodgerblue", lw=3, label=label)
    ax2.set_xlabel(r"$A_{br}$ $[m^2]$")
    ax2.set_ylabel(r"$p(A_{br})$ $[-]$")
    ax2.set_xlim(0, 10)
    lg = ax2.legend(loc=1, fontsize=10)
    lg.get_frame().set_color("w")

    # major axis
    xpdf = np.arange(0., 6, 0.01)
    pars = stats.beta.fit(df["LA_max"] / df["LB_max"])
    ypdf = stats.beta.pdf(xpdf, *pars)
    kde = sm.nonparametric.KDEUnivariate(df["LA_max"] / df["LB_max"])
    kde.fit()
    mode = kde.support[np.argmax(kde.density)]
    label = r"Beta PDF"

    ax3.hist(df["LA_max"] / df["LB_max"], density=True,
             bins=np.arange(0, 5, 0.25), color="0.15", alpha=0.5)
    ax3.axvline((df["LA_max"] / df["LB_max"]).mean(), color="indigo", lw=3,
                ls="--", label="Mean={0:.1f}".format((df["LA_max"] / df["LB_max"]).mean()))
    ax3.axvline(mode, color="coral", lw=3, ls="--",
                label="Mode={0:.1f}".format(mode))
    ax3.plot(xpdf, ypdf, color="dodgerblue", lw=3, label=label)
    ax3.set_xlabel(r"$a/b$ $[-]$")
    ax3.set_ylabel(r"$p(a/b)$ $[-]$")
    ax3.set_xlim(0, 6)
    lg = ax3.legend(loc=1, fontsize=10)
    lg.get_frame().set_color("w")

    # Duncan's
    xpdf = np.arange(0., 10, 0.1)
    pars = stats.beta.fit(df["Ab"] / df["LB_max"]**2)
    ypdf = stats.beta.pdf(xpdf, *pars)
    kde = sm.nonparametric.KDEUnivariate(df["Ab"] / df["LB_max"]**2)
    kde.fit()
    mode = kde.support[np.argmax(kde.density)]
    label = r"Beta PDF"

    ax4.hist(df["Ab"] / df["LB_max"]**2, density=True,
             color="0.15", alpha=0.5, bins=np.arange(0, 10, 0.5))
    ax4.plot(xpdf, ypdf, color="dodgerblue", lw=3, label=label)
    ax4.axvline((df["LA_max"] / df["LB_max"]).mean(), color="indigo", lw=3,
                ls="--", label="Mean={0:.1f}".format((df["LA_max"] / df["LB_max"]).mean()))
    ax4.axvline(mode, color="coral", lw=3, ls="--",
                label="Mode={0:.1f}".format(mode))
    ax4.axvline(0.1, color="r", lw=3, ls="--", label="Duncan (1981)")
    ax4.set_xlabel(r"$A_{{br}}/b^{2}$ $[-]$")
    ax4.set_ylabel(r"$p(A_{{br}}/b^{2})$ $[-]$")
    ax4.set_xlim(0, 10)
    lg = ax4.legend(loc=1, fontsize=10)
    lg.get_frame().set_color("w")

    # Area over time
    model = Pipeline(steps=[("poly",
                             PolynomialFeatures(degree=2,
                                                interaction_only=False,
                                                include_bias=True)),
                            ("reg",
                             LinearRegression(fit_intercept=True))])
    model.fit(df["DT"].values.reshape(-1, 1), df["Ab"])
    ytrue = np.squeeze(np.array(df["Ab"].values.reshape(-1, 1)))
    ypred = np.squeeze(model.predict(df["DT"].values.reshape(-1, 1)))
    r, p = stats.pearsonr(ytrue, ypred)

    i = model.steps[1][1].intercept_
    c = model.steps[1][1].coef_

    text = r"$A_{{br}} = {} + {}T_{{br}} + {}T_{{br}}^2$".format(round(i, 2),
                                                                 round(
                                                                     c[1], 2),
                                                                 round(c[2], 2))

    ax5.scatter(df["DT"], df["Ab"], color="k", zorder=2)
    sns.regplot(df["DT"], df["Ab"], x_bins=np.arange(0, 2, 0.1), order=2,
                color="dodgerblue", ax=ax5, )
    ax5.text(0.95, 0.95, text, fontsize=9,
             va="top", zorder=100, transform=ax5.transAxes, ha="right",
             bbox=dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))
    ax5.text(0.95, 0.825, r"$r_{{xy}}={}, p<<0.05$".format(round(r, 2)), fontsize=9,
             va="top", zorder=100, transform=ax5.transAxes, ha="right",
             bbox=dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))
    ax5.set_xlabel(r"$T_{br}$ $[s]$")
    ax5.set_ylabel(r"$A_{br}$ $[m^2]$")

    # Lambda
    ax6.plot(cb, cb**-6, color="k", ls="--", lw=3, label="Phillips (1985)")
    for wind, _Lambda in zip(target_winds, Lambda):

        Lambda_s = running_mean(_Lambda, 10)
        Lambda_s[Lambda_s < 10E-6] = np.ma.masked

        color = cmap(norm(wind))
        ax6.scatter(cb, _Lambda, 60, alpha=0.5, facecolor=color, edgecolor="w")
        ax6.plot(cb, Lambda_s, lw=3, zorder=20, color=color,
                 label=r"$U_{{10}}={}$".format(wind))
    ax6.set_yscale("log")
    ax6.set_ylim(10E-5, 10 * np.max(Lambda))
    ax6.set_xlim(1, 10)
    lg = ax6.legend(loc=1, fontsize=9)
    lg.get_frame().set_color("w")
    ax6.set_xlabel(r"$c$ $[ms^{-1}]$")
    ax6.set_ylabel(r"$\Lambda(c)$ $[m^{-2}s]$")

    # format axes
    k = 0
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:

        ax.grid(color="w", lw=2, ls="-")
        sns.despine(ax=ax)
        ax.set_xlim(0)

        ax.text(0.05, 0.95, ascii_lowercase[k] + ")",
                va="top", zorder=100, transform=ax.transAxes, ha="left",
                bbox=dict(
                    boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7))

        k += 1

    fig.tight_layout()
    plt.savefig(out, dpi=300, pad_inches=0.1, bbox_inches='tight')
    plt.show()
