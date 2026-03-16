"""
Autocorrelation Test
Measures the linear dependency between a sequence and its lagged versions.
True random sequences should have autocorrelation ≈ 0 for all non-zero lags.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import List


@dataclass
class AutocorrelationResult:
    max_lag: int
    acf_values: np.ndarray          # ACF for lags 1 … max_lag
    significant_lags: List[int]     # Lags where |ACF| exceeds confidence band
    max_acf: float                  # Largest absolute ACF value observed
    is_random: bool                 # True → no significant autocorrelation
    verdict: str

    def __str__(self) -> str:
        sig = self.significant_lags if self.significant_lags else "none"
        return (
            f"Autocorrelation Test (max_lag={self.max_lag})\n"
            f"  Max |ACF|          : {self.max_acf:.6f}\n"
            f"  Significant lags   : {sig}\n"
            f"  Verdict            : {self.verdict}"
        )


def compute_acf(data: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute the normalized Autocorrelation Function (ACF) for lags 1 … max_lag.

    Uses the unbiased estimator:
        ACF(k) = cov(X_t, X_{t+k}) / var(X)

    Args:
        data:    1-D numeric array.
        max_lag: Maximum lag to compute.

    Returns:
        Array of ACF values, shape (max_lag,), for lags [1, 2, ..., max_lag].
    """
    x = data.astype(np.float64)
    x -= x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag)

    acf = np.array([
        np.dot(x[:len(x) - k], x[k:]) / ((len(x) - k) * var)
        for k in range(1, max_lag + 1)
    ])
    return acf


def autocorrelation_test(data: np.ndarray, max_lag: int = 50,
                          alpha: float = 0.05) -> AutocorrelationResult:
    """
    Test a sequence for autocorrelation up to `max_lag`.

    The 95 % confidence band for a white-noise process is ±z / √n,
    where z is the quantile of the standard normal.

    Args:
        data:    1-D numeric array.
        max_lag: How many lags to examine.
        alpha:   Significance level for the confidence band.

    Returns:
        AutocorrelationResult dataclass.
    """
    n = len(data)
    acf = compute_acf(data, max_lag)

    # Two-sided critical value
    z_crit = norm.ppf(1 - alpha / 2)
    confidence_band = z_crit / np.sqrt(n)

    significant_lags = [lag + 1 for lag, val in enumerate(acf)
                        if abs(val) > confidence_band]
    max_acf = float(np.max(np.abs(acf)))
    is_random = len(significant_lags) == 0

    if is_random:
        verdict = (
            f"PASS — No lags exceed the {(1-alpha)*100:.0f}% confidence band "
            f"(±{confidence_band:.4f}). Sequence appears uncorrelated."
        )
    else:
        verdict = (
            f"FAIL — {len(significant_lags)} lag(s) exceed the confidence band "
            f"(±{confidence_band:.4f}): {significant_lags[:10]}{'...' if len(significant_lags) > 10 else ''}. "
            f"Autocorrelation detected — generator may be predictable."
        )

    return AutocorrelationResult(
        max_lag=max_lag,
        acf_values=acf,
        significant_lags=significant_lags,
        max_acf=max_acf,
        is_random=is_random,
        verdict=verdict,
    )


def runs_test(data: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Wald–Wolfowitz runs test for independence.
    A 'run' is a maximal non-empty segment of identical values relative
    to the median.  Too few or too many runs both indicate non-randomness.

    Args:
        data:  1-D numeric array.
        alpha: Significance level.

    Returns:
        Dictionary with keys: runs, expected_runs, z_stat, p_value, is_random, verdict.
    """
    x = data.astype(np.float64)
    median = np.median(x)
    binary = (x > median).astype(int)

    n1 = binary.sum()          # count above median
    n2 = len(binary) - n1      # count at/below median

    runs = 1 + np.sum(np.diff(binary) != 0)
    expected = (2 * n1 * n2) / (n1 + n2) + 1
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / \
               ((n1 + n2) ** 2 * (n1 + n2 - 1))

    if variance == 0:
        z_stat, p_value = 0.0, 1.0
    else:
        z_stat = (runs - expected) / np.sqrt(variance)
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))   # two-tailed

    is_random = p_value >= alpha
    verdict = (
        f"PASS — p={p_value:.4f} ≥ α={alpha}: run count ({int(runs)}) consistent with independence."
        if is_random else
        f"FAIL — p={p_value:.4f} < α={alpha}: run count ({int(runs)}) significantly "
        f"{'too few' if runs < expected else 'too many'} (expected {expected:.1f})."
    )

    return dict(runs=int(runs), expected_runs=expected,
                z_stat=z_stat, p_value=p_value,
                is_random=is_random, verdict=verdict)


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Truly random ===")
    rand_data = rng.integers(0, 256, size=5000).astype(np.float64)
    result = autocorrelation_test(rand_data)
    print(result)

    print("\n=== LCG (periodic) ===")
    a, c, m = 1664525, 1013904223, 2**32
    x = 42
    lcg = []
    for _ in range(5000):
        x = (a * x + c) % m
        lcg.append(x % 256)
    lcg_data = np.array(lcg, dtype=np.float64)
    result_lcg = autocorrelation_test(lcg_data)
    print(result_lcg)

    print("\n=== Runs test ===")
    print(runs_test(rand_data))
