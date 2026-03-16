"""
Distribution Visualization
Plots frequency histograms and comparative distribution charts
for all entropy sources under evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Optional


FIGURES_DIR = Path.cwd() / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# House style
PALETTE = {
    "Pseudo RNG":      "#4C72B0",
    "Simulated QRNG":  "#DD8452",
    "ANU QRNG":        "#55A868",
    "Biased RNG":      "#C44E52",
    "LCG":             "#8172B2",
}
DEFAULT_COLORS = list(PALETTE.values())


def _source_color(name: str, idx: int) -> str:
    return PALETTE.get(name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


# ── Single-source histogram ──────────────────────────────────────────────────

def plot_histogram(data: np.ndarray, label: str = "Source",
                   bins: int = 64, save: bool = True,
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot a frequency histogram for a single entropy source.

    Args:
        data:  Numeric array (uint8 or normalized float).
        label: Title label for the source.
        bins:  Number of histogram bins.
        save:  Whether to save the figure to results/figures/.
        ax:    Optional existing Axes to draw on.

    Returns:
        Matplotlib Figure object.
    """
    standalone = ax is None
    fig = None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    color = PALETTE.get(label, DEFAULT_COLORS[0])
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.4, alpha=0.85)

    # Ideal uniform reference line
    expected_height = len(data) / bins
    ax.axhline(expected_height, color="red", linestyle="--", linewidth=1.2,
               label="Expected (uniform)")

    ax.set_title(f"Distribution — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        fig.tight_layout()
        if save:
            path = FIGURES_DIR / f"histogram_{label.replace(' ', '_').lower()}.png"
            fig.savefig(path, dpi=150)
            print(f"[Viz] Saved: {path}")
    return fig


# ── Multi-source comparison ──────────────────────────────────────────────────

def plot_distribution_comparison(sources: Dict[str, np.ndarray],
                                   bins: int = 64,
                                   save: bool = True) -> plt.Figure:
    """
    Plot side-by-side histograms for multiple entropy sources.

    Args:
        sources: Mapping of {source_name: data_array}.
        bins:    Number of histogram bins.
        save:    Whether to save the figure.

    Returns:
        Matplotlib Figure.
    """
    n = len(sources)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(sources.items()):
        plot_histogram(data, label=name, bins=bins, save=False, ax=axes[idx])

    fig.suptitle("Entropy Source Distribution Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "distribution_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {path}")
    return fig


def plot_overlay_kde(sources: Dict[str, np.ndarray],
                      save: bool = True) -> plt.Figure:
    """
    Overlay KDE-smoothed density curves for all sources on a single plot.
    Useful for a quick visual comparison of distribution shapes.

    Args:
        sources: Mapping of {source_name: data_array}.
        save:    Whether to save the figure.

    Returns:
        Matplotlib Figure.
    """
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (name, data) in enumerate(sources.items()):
        color = _source_color(name, idx)
        sample = data.astype(np.float64)
        kde = gaussian_kde(sample, bw_method="silverman")
        xs = np.linspace(sample.min(), sample.max(), 500)
        ax.plot(xs, kde(xs), color=color, linewidth=2.0, label=name)
        ax.fill_between(xs, kde(xs), alpha=0.12, color=color)

    ax.set_title("Smoothed Density Comparison (KDE)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "kde_comparison.png"
        fig.savefig(path, dpi=150)
        print(f"[Viz] Saved: {path}")
    return fig


def plot_qq(data: np.ndarray, label: str = "Source", save: bool = True) -> plt.Figure:
    """
    Quantile-Quantile plot against an ideal uniform distribution.
    Points should lie on the diagonal for a truly uniform source.
    """
    from scipy.stats import uniform as uniform_dist

    fig, ax = plt.subplots(figsize=(5, 5))
    n = len(data)
    empirical = np.sort(data.astype(np.float64))
    # Normalize to [0, 1]
    empirical = (empirical - empirical.min()) / max(empirical.max() - empirical.min(), 1e-9)
    theoretical = uniform_dist.ppf(np.linspace(0, 1, n, endpoint=False) + 0.5 / n)

    ax.scatter(theoretical, empirical, s=3, alpha=0.4,
               color=PALETTE.get(label, DEFAULT_COLORS[0]))
    ax.plot([0, 1], [0, 1], "r--", linewidth=1.5, label="Ideal uniform")
    ax.set_title(f"Q-Q Plot — {label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Theoretical quantiles", fontsize=11)
    ax.set_ylabel("Empirical quantiles", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / f"qq_{label.replace(' ', '_').lower()}.png"
        fig.savefig(path, dpi=150)
        print(f"[Viz] Saved: {path}")
    return fig


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sources = {
        "Pseudo RNG": rng.integers(0, 256, 5000, dtype=np.uint8),
        "Biased RNG": np.array([rng.choice(256, p=np.where(
            np.arange(256) == 42, 0.3, 0.7/255)) for _ in range(5000)], dtype=np.uint8),
    }
    plot_distribution_comparison(sources)
    plot_overlay_kde(sources)
    print("Plots saved.")
