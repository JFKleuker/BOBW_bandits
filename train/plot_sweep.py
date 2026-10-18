"""Reads the CSV written by `sweep.py` and plots aggregate regret.

Regret is first averaged over train seeds per environment, then mean and standard error are
taken over environments, so each environment counts equally.

Usage:
    python train/plot_sweep.py                       # reads results/sweep.csv
    python train/plot_sweep.py --csv path/to/sweep.csv --out-dir results/plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

# Fixed categorical order: an algorithm keeps its color in every figure.
COLORS = {"FTL": "#2a78d6", "SAPO": "#eb6834", "FTRLBOBW": "#1baf7a"}
FALLBACK_COLORS = ["#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(csv: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv)


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean regret per environment (over seeds), then mean / std. error over environments."""
    per_env = df.groupby(by + ["env_id"], as_index=False)["regret"].mean()
    agg = per_env.groupby(by)["regret"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    return agg


def color_of(algorithm: str, algorithms: list[str]) -> str:
    if algorithm in COLORS:
        return COLORS[algorithm]
    others = [a for a in algorithms if a not in COLORS]
    return FALLBACK_COLORS[others.index(algorithm) % len(FALLBACK_COLORS)]


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def add_title_and_legend(fig, axes, title: str):
    """Title on top, one legend row below the panels (handles collected from all panels).
    Margins are set in inches so the layout works for any figure height."""
    handles = {}
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, h)
    height = fig.get_figheight()
    fig.suptitle(title, y=1 - 0.1 / height, va="top", color=INK, fontsize=11)
    fig.legend(handles.values(), handles.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.35 / height, 1, 1 - 0.4 / height))


def plot_line(ax, data, x, algorithms):
    for algorithm in algorithms:
        d = data[data["algorithm"] == algorithm].sort_values(x)
        if d.empty:
            continue
        color = color_of(algorithm, algorithms)
        ax.fill_between(d[x], d["mean"] - d["sem"], d["mean"] + d["sem"], color=color, alpha=0.15, lw=0)
        ax.plot(d[x], d["mean"], color=color, lw=2, marker="o", ms=5, label=algorithm)


def plot_regret_vs_horizon(df: pd.DataFrame, out: Path):
    """Small multiples: one panel per K, regret against num_steps (log-log)."""
    agg = aggregate(df, ["num_arms", "algorithm", "num_steps"])
    algorithms = list(df["algorithm"].unique())
    arms = sorted(df["num_arms"].unique())
    ncols = min(3, len(arms))
    nrows = int(np.ceil(len(arms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), squeeze=False)
    for ax, num_arms in zip(axes.flat, arms):
        plot_line(ax, agg[agg["num_arms"] == num_arms], "num_steps", algorithms)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"K = {num_arms}", fontsize=10, color=INK, loc="left")
        style_axes(ax)
    for ax in axes.flat[len(arms):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("num_steps", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("pseudo-regret", color=MUTED, fontsize=9)
    add_title_and_legend(fig, axes, "Pseudo-regret vs. horizon (mean ± s.e. over environments)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regret_vs_arms(df: pd.DataFrame, out: Path):
    """One panel per horizon, regret against K."""
    agg = aggregate(df, ["num_steps", "algorithm", "num_arms"])
    algorithms = list(df["algorithm"].unique())
    horizons = sorted(df["num_steps"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(3.6 * len(horizons), 3.0), squeeze=False)
    for ax, num_steps in zip(axes[0], horizons):
        plot_line(ax, agg[agg["num_steps"] == num_steps], "num_arms", algorithms)
        ax.set_title(f"num_steps = {num_steps:,}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("K (number of arms)", color=MUTED, fontsize=9)
        style_axes(ax)
    axes[0, 0].set_ylabel("pseudo-regret", color=MUTED, fontsize=9)
    add_title_and_legend(fig, axes, "Pseudo-regret vs. number of arms (mean ± s.e. over environments)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean regret per (algorithm, num_steps), averaged over the environments of the K values
    that every algorithm was run on (e.g. FTRLBOBW skips K=2)."""
    arms_per_algorithm = df.groupby("algorithm")["num_arms"].unique()
    common = set.intersection(*(set(a) for a in arms_per_algorithm))
    agg = aggregate(df[df["num_arms"].isin(common)], ["algorithm", "num_steps"])
    return agg.pivot(index="algorithm", columns="num_steps", values="mean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", default="results/sweep.csv")
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()

    df = load(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(df):,} rows, {df['env_id'].nunique()} environments, "
          f"{df['train_seed'].nunique()} seeds")
    print("\nMean pseudo-regret over all environments:")
    print(summary_table(df).round(1).to_string())

    plot_regret_vs_horizon(df, out_dir / "regret_vs_horizon.png")
    plot_regret_vs_arms(df, out_dir / "regret_vs_arms.png")
    print(f"\nPlots written to {out_dir}/")
