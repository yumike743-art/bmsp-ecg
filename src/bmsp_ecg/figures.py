"""Report-sized vector figures with reusable source tables in the run directory."""

import json

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
import yaml

from .signal import BRANCH_LABELS

TEAL = "#126659"
ORANGE = "#9B4D28"
INK = "#152C3A"
GREY = "#526770"
GRID = "#DBE4E1"
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 8.3, "axes.labelsize": 8.3,
    "axes.titlesize": 9.2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": GRID, "axes.titleweight": "bold",
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "axes.facecolor": "white",
}


def save(fig, folder, name):
    for extension in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{extension}", dpi=240)
    plt.close(fig)


def robustness(run, folder):
    table = pd.read_csv(run / "noise_metrics.csv", dtype={"snr_db": str}, keep_default_na=False)
    clean = table.query("noise == 'clean'")
    all_scores = table[["raw_f1", "filtered_f1"]].to_numpy()
    lo = max(0, np.floor((all_scores.min() - .04) * 10) / 10)
    hi = min(1, np.ceil((all_scores.max() + .04) * 10) / 10)
    lims = table[["lo", "hi"]].to_numpy()
    delta_lo = np.floor((min(lims.min(), 0) - .005) * 40) / 40
    delta_hi = np.ceil((max(lims.max(), 0) + .005) * 40) / 40
    fig, axs = plt.subplots(2, 2, figsize=(7.02, 3.66), sharex="col", sharey="row",
                            gridspec_kw={"height_ratios": [1.45, 1]})
    fig.subplots_adjust(left=.104, right=.986, bottom=.14, top=.81, wspace=.14, hspace=.24)
    for j, (noise, title) in enumerate([("white", "A  White Gaussian noise"), ("drift", "B  0.33 Hz baseline drift")]):
        subset = pd.concat([clean, table.query("noise == @noise").sort_values(
            "snr_db", key=lambda values: values.astype(int), ascending=False,
        )], ignore_index=True)
        x = np.arange(len(subset))
        top, bottom = axs[:, j]
        styles = [
            (subset.raw_f1, BRANCH_LABELS["raw"],
             dict(color=ORANGE, marker="s", mfc="white", lw=1.35, ms=4.5, ls="--")),
            (subset.filtered_f1, BRANCH_LABELS["filtered"],
             dict(color=TEAL, marker="o", lw=1.65, ms=4.6, ls="-")),
        ]
        for values, label, style in styles:
            top.plot(x[1:], values.iloc[1:], label=label, **style)
            top.plot(x[:1], values.iloc[:1], **style)
        top.set_title(title, loc="left", pad=8)
        top.set_ylim(lo, hi)
        bottom.axhline(0, color=GREY, lw=.8, ls=":")
        bottom.vlines(x, subset.lo, subset.hi, color=INK, lw=1.0)
        bottom.hlines(subset.lo, x - .035, x + .035, color=INK, lw=1.0)
        bottom.hlines(subset.hi, x - .035, x + .035, color=INK, lw=1.0)
        bottom.plot(x, subset.delta, "D", ms=3.8, color=INK)
        bottom.set_ylim(delta_lo, delta_hi)
        bottom.yaxis.set_major_locator(MultipleLocator(.025))
        bottom.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        bottom.set_xticks(x, ["No added\nnoise", *subset.snr_db.iloc[1:]])
        bottom.set_xlabel("Injected SNR (dB)", labelpad=3.5)
        for ax in (top, bottom):
            ax.yaxis.grid(True, color=GRID, lw=.65, zorder=0)
            ax.axvline(.5, color=GRID, lw=.8, ls=":")
            ax.set_xlim(-.29, len(x) - .71)
            ax.tick_params(axis="both", length=2.5, color=GRID)
    axs[0, 0].set_ylabel("Macro-F1")
    axs[1, 0].set_ylabel("Δ Macro-F1")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.55, .99), ncols=2)
    save(fig, folder, "fig_robustness")


def waveform(run, folder, name):
    wave = pd.read_csv(run / f"{name}_waveform.csv")
    refs = pd.read_csv(run / f"{name}_reference.csv")
    meta = json.loads((run / f"{name}_meta.json").read_text())
    events = pd.read_csv(run / f"{name}_events.csv") if name == "failure" else None
    fs = meta["fs"]
    is_failure = events is not None
    tolerance = yaml.safe_load((run / "experiment.yaml").read_text())["detection"]["tolerance_ms"] / 1000
    height = 2.58 if is_failure else 2.29
    fig, axs = plt.subplots(2, 1, figsize=(7.02, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.976, bottom=.19, top=.826 if is_failure else .87, hspace=.46)
    focus = meta["focus_sample"] / fs
    limits = wave[["raw_mV", "filtered_mV"]].to_numpy()
    low, high = limits.min(), limits.max()
    room = max(.1, .18 * (high - low))
    for ax, label, title, color in zip(
        axs, ["raw_mV", "filtered_mV"],
        [f"A  {BRANCH_LABELS['raw']}", f"B  {BRANCH_LABELS['filtered']}"], [ORANGE, TEAL],
    ):
        ax.yaxis.grid(True, color=GRID, lw=.6, zorder=0)
        if is_failure:
            ax.axvspan(focus - tolerance, focus + tolerance, color="#ECE0CA", alpha=.83, lw=0)
        ax.plot(wave.time_s, wave[label], color=color, lw=1.35, zorder=3)
        for ref in refs.itertuples(index=False):
            ax.axvline(ref.sample / fs, color=GREY, ls=(0, (2, 2)), lw=.85, alpha=.9, zorder=2)
        ax.set_title(title, loc="left", pad=4.3)
        ax.set_ylim(low - room * .65, high + room * 1.3)
        ax.set_xlim(meta["start"] / fs, meta["stop"] / fs)
        ax.tick_params(axis="both", length=2.5, color=GRID)
    if is_failure:
        labels = refs[refs.label.isin(["N", "S", "V", "F"])]
        for ref in labels.itertuples(index=False):
            x = ref.sample / fs
            missed = ref.sample == meta["focus_sample"] and meta["focus_status"] == "miss"
            axs[0].text(x, high + room * .74, f"{ref.label} missed" if missed else ref.label,
                        ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if missed else "normal")
        detections = events.dropna(subset=["det_sample"])
        indices = detections.det_sample.to_numpy(dtype=int) - meta["start"]
        within = (indices >= 0) & (indices < len(wave))
        axs[1].scatter(detections.det_sample[within] / fs, wave.filtered_mV.iloc[indices[within]],
                       marker="v", s=27, c=TEAL, edgecolors="white", linewidths=.55, zorder=4)
    axs[1].set_xlabel("Time from record start (s)", labelpad=4)
    fig.supylabel("Amplitude (mV)", x=.025, fontsize=8.3)
    save(fig, folder, f"fig_{name}")


def build_figures(run, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(STYLE):
        robustness(run, folder)
        waveform(run, folder, "qc")
        waveform(run, folder, "failure")
