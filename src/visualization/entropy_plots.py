"""
Entropy & Autocorrelation Visualization
Bar charts, heat-maps, and line plots for entropy metrics and ACF.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Optional


FIGURES_DIR = Path.cwd() / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "Pseudo RNG":      "#4C72B0",
    "Simulated QRNG":  "#DD8452",
    "ANU QRNG":        "#55A868",
    "Biased RNG":      "#C44E52",
    "LCG":             "#8172B2",
}
DEFAULT_COLORS = list(PALETTE.values())


def _color(name: str, idx: int) -> str:
    return PALETTE.get(name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])


# ── Entropy bar chart ────────────────────────────────────────────────────────

def plot_entropy_comparison(shannon_values: Dict[str, float],
                             min_entropy_values: Optional[Dict[str, float]] = None,
                             save: bool = True) -> plt.Figure:
    """
    Grouped bar chart comparing Shannon entropy (and optionally min-entropy)
    across multiple entropy sources.

    Args:
        shannon_values:    {source_name: Shannon entropy in bits}.
        min_entropy_values:{source_name: min-entropy in bits} (optional).
        save:              Save to results/figures/.

    Returns:
        Matplotlib Figure.
    """
    names  = list(shannon_values.keys())
    h_vals = [shannon_values[n] for n in names]
    max_h  = np.log2(256)   # 8 bits theoretical maximum

    n = len(names)
    x = np.arange(n)
    width = 0.35 if min_entropy_values else 0.55

    fig, ax = plt.subplots(figsize=(max(7, 2 * n), 5))
    colors = [_color(nm, i) for i, nm in enumerate(names)]

    bars_h = ax.bar(x - (width / 2 if min_entropy_values else 0),
                     h_vals, width, label="Shannon H(X)", color=colors, alpha=0.88,
                     edgecolor="white")

    if min_entropy_values:
        hmin_vals = [min_entropy_values.get(n, 0) for n in names]
        bars_hmin = ax.bar(x + width / 2, hmin_vals, width,
                            label="Min-Entropy Hmin", color=colors, alpha=0.50,
                            edgecolor="white", hatch="//")
        for bar, val in zip(bars_hmin, hmin_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Value labels on Shannon bars
    for bar, val in zip(bars_h, h_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Theoretical maximum reference
    ax.axhline(max_h, color="crimson", linestyle="--", linewidth=1.3,
               label=f"Max entropy ({max_h:.1f} bits)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, max_h * 1.15)
    ax.set_ylabel("Entropy (bits / symbol)", fontsize=11)
    ax.set_title("Entropy Comparison Across Sources", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "entropy_comparison.png"
        fig.savefig(path, dpi=150)
        print(f"[Viz] Saved: {path}")
    return fig


# ── Block entropy time-series ────────────────────────────────────────────────

def plot_block_entropy(block_entropies: Dict[str, np.ndarray],
                        block_size: int = 256,
                        save: bool = True) -> plt.Figure:
    """
    Line plot of Shannon entropy computed over rolling blocks.
    Useful for detecting time-varying entropy (non-stationarity).

    Args:
        block_entropies: {source_name: array of per-block entropy values}.
        block_size:      Block size used during computation (for axis labels).
        save:            Save figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    max_h = np.log2(256)

    for idx, (name, values) in enumerate(block_entropies.items()):
        color = _color(name, idx)
        ax.plot(values, color=color, linewidth=1.5, alpha=0.85, label=name)

    ax.axhline(max_h, color="crimson", linestyle="--", linewidth=1.0,
               label=f"Max H ({max_h:.1f} bits)")
    ax.set_title(f"Block Entropy Over Time (block_size={block_size})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Block index", fontsize=11)
    ax.set_ylabel("Shannon H (bits)", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "block_entropy_timeseries.png"
        fig.savefig(path, dpi=150)
        print(f"[Viz] Saved: {path}")
    return fig


# ── Autocorrelation plot ─────────────────────────────────────────────────────

def plot_autocorrelation(acf_data: Dict[str, np.ndarray],
                          n_samples: Dict[str, int],
                          alpha: float = 0.05,
                          save: bool = True) -> plt.Figure:
    """
    Stem + line ACF plot for one or more entropy sources.
    Confidence bands are drawn for each source based on its sample size.

    Args:
        acf_data:  {source_name: ACF array (lags 1 … max_lag)}.
        n_samples: {source_name: number of data points used}.
        alpha:     Significance level for confidence bands.
        save:      Save figure.

    Returns:
        Matplotlib Figure.
    """
    from scipy.stats import norm

    n_sources = len(acf_data)
    fig, axes = plt.subplots(n_sources, 1,
                              figsize=(10, 3.5 * n_sources),
                              squeeze=False)

    for idx, (name, acf) in enumerate(acf_data.items()):
        ax = axes[idx][0]
        lags = np.arange(1, len(acf) + 1)
        color = _color(name, idx)

        z_crit = norm.ppf(1 - alpha / 2)
        n = n_samples.get(name, 1000)
        band = z_crit / np.sqrt(n)

        markerline, stemlines, baseline = ax.stem(lags, acf, linefmt=color,
                                                   markerfmt="o", basefmt="k-")
        plt.setp(markerline, color=color, markersize=3)
        plt.setp(stemlines, linewidth=0.8, alpha=0.7)

        ax.axhline(band,  color="crimson", linestyle="--", linewidth=1.0,
                   label=f"±{band:.4f} ({(1-alpha)*100:.0f}% CI)")
        ax.axhline(-band, color="crimson", linestyle="--", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.6)

        ax.set_title(f"Autocorrelation — {name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Lag", fontsize=10)
        ax.set_ylabel("ACF", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "autocorrelation_plots.png"
        fig.savefig(path, dpi=150)
        print(f"[Viz] Saved: {path}")
    return fig


def plot_heatmap_acf_matrix(sources: Dict[str, np.ndarray],
                              max_lag: int = 30,
                              save: bool = True) -> plt.Figure:
    """
    Heat-map of ACF values for all sources vs lags.
    Lets you visually compare correlation profiles side by side.

    Args:
        sources:  {source_name: raw data array}.
        max_lag:  Number of lags to include in the heat-map.
        save:     Save figure.

    Returns:
        Matplotlib Figure.
    """
    from ..tests.autocorrelation import compute_acf

    names = list(sources.keys())
    matrix = np.stack([compute_acf(v, max_lag) for v in sources.values()])

    fig, ax = plt.subplots(figsize=(12, max(3, len(names) * 0.8 + 1)))
    im = ax.imshow(np.abs(matrix), aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.1)
    plt.colorbar(im, ax=ax, label="|ACF|")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Lag", fontsize=11)
    ax.set_title(f"ACF Magnitude Heat-map (max_lag={max_lag})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        path = FIGURES_DIR / "acf_heatmap.png"
        fig.savefig(path, dpi=150)
        print(f"[Viz] Saved: {path}")
    return fig


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h_vals = {"Pseudo RNG": 7.95, "Simulated QRNG": 7.98, "ANU QRNG": 7.99, "Biased RNG": 5.12}
    hmin   = {"Pseudo RNG": 7.90, "Simulated QRNG": 7.93, "ANU QRNG": 7.97, "Biased RNG": 1.74}
    plot_entropy_comparison(h_vals, hmin)
    print("Plots saved.")
