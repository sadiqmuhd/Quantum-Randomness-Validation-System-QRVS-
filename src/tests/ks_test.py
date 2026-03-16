"""
Kolmogorov–Smirnov Uniformity Test
Compares the empirical CDF of a sample against an ideal uniform distribution
using the two-sided KS statistic.
"""

import numpy as np
from scipy.stats import kstest, ks_2samp
from dataclasses import dataclass


@dataclass
class KSResult:
    statistic: float      # Maximum CDF deviation (D statistic)
    p_value: float
    is_random: bool       # True → fail to reject H₀
    verdict: str

    def __str__(self) -> str:
        return (
            f"Kolmogorov–Smirnov Test\n"
            f"  D statistic : {self.statistic:.6f}\n"
            f"  p-value     : {self.p_value:.4f}\n"
            f"  Verdict     : {self.verdict}"
        )


def ks_test_uniform(data: np.ndarray, alpha: float = 0.05) -> KSResult:
    """
    One-sample KS test comparing data to a Uniform(0, 1) distribution.

    If data is integer (e.g. uint8), it is normalized to [0, 1] before testing.

    Args:
        data:  NumPy array of random values.
        alpha: Significance level.

    Returns:
        KSResult dataclass.
    """
    # Normalize integers to [0, 1]
    if np.issubdtype(data.dtype, np.integer):
        lo, hi = float(data.min()), float(data.max())
        if hi == lo:
            hi = lo + 1.0
        sample = (data.astype(np.float64) - lo) / (hi - lo)
    else:
        sample = data.astype(np.float64)

    stat, p_value = kstest(sample, "uniform")
    is_random = p_value >= alpha
    verdict = (
        f"PASS — p={p_value:.4f} ≥ α={alpha}: distribution consistent with uniform."
        if is_random else
        f"FAIL — p={p_value:.4f} < α={alpha}: significant deviation from uniform CDF."
    )

    return KSResult(
        statistic=float(stat),
        p_value=float(p_value),
        is_random=is_random,
        verdict=verdict,
    )


def ks_test_two_sample(data_a: np.ndarray, data_b: np.ndarray,
                        label_a: str = "Source A", label_b: str = "Source B",
                        alpha: float = 0.05) -> KSResult:
    """
    Two-sample KS test to determine if two entropy sources have the same
    underlying distribution.

    Args:
        data_a:  First random sequence.
        data_b:  Second random sequence.
        label_a: Human-readable name for source A.
        label_b: Human-readable name for source B.
        alpha:   Significance level.

    Returns:
        KSResult dataclass.
    """
    a = data_a.astype(np.float64)
    b = data_b.astype(np.float64)

    stat, p_value = ks_2samp(a, b)
    # High p → cannot distinguish the two distributions
    same_dist = p_value >= alpha
    verdict = (
        f"SIMILAR — p={p_value:.4f} ≥ α={alpha}: {label_a} and {label_b} "
        f"appear to share the same distribution."
        if same_dist else
        f"DIFFERENT — p={p_value:.4f} < α={alpha}: {label_a} and {label_b} "
        f"have significantly different distributions."
    )

    return KSResult(
        statistic=float(stat),
        p_value=float(p_value),
        is_random=same_dist,
        verdict=verdict,
    )


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Uniform uint8 ===")
    uniform = rng.integers(0, 256, size=5000, dtype=np.uint8)
    print(ks_test_uniform(uniform))

    print("\n=== Gaussian (should fail) ===")
    gaussian = rng.normal(127.5, 40, size=5000).astype(np.uint8)
    print(ks_test_uniform(gaussian))

    print("\n=== Two-sample comparison ===")
    sample_a = rng.integers(0, 256, size=2000, dtype=np.uint8)
    sample_b = rng.integers(0, 256, size=2000, dtype=np.uint8)
    print(ks_test_two_sample(sample_a, sample_b, "PRNG-A", "PRNG-B"))
