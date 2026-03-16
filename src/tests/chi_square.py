"""
Chi-Square Uniformity Test
Tests whether an observed frequency distribution matches the expected
uniform distribution for a random sequence.
"""

import numpy as np
from scipy.stats import chisquare
from dataclasses import dataclass


@dataclass
class ChiSquareResult:
    statistic: float
    p_value: float
    degrees_of_freedom: int
    is_random: bool          # True  → fail to reject H₀ (looks random)
    verdict: str

    def __str__(self) -> str:
        return (
            f"Chi-Square Test\n"
            f"  Statistic : {self.statistic:.4f}\n"
            f"  p-value   : {self.p_value:.4f}\n"
            f"  df        : {self.degrees_of_freedom}\n"
            f"  Verdict   : {self.verdict}"
        )


def chi_square_test(data: np.ndarray, bins: int = 256,
                    alpha: float = 0.05) -> ChiSquareResult:
    """
    Perform a chi-square goodness-of-fit test for uniformity.

    The null hypothesis H₀ is that the data is uniformly distributed.

    Args:
        data:  Integer or float array. If float, values are assumed in [0, 1]
               and are scaled to [0, bins-1].
        bins:  Number of bins (categories). Default 256 matches uint8 range.
        alpha: Significance level for the accept/reject decision.

    Returns:
        ChiSquareResult dataclass.
    """
    # Bin the data
    if np.issubdtype(data.dtype, np.floating):
        indices = (data * bins).astype(int)
        indices = np.clip(indices, 0, bins - 1)
    else:
        indices = data.astype(int) % bins

    observed, _ = np.histogram(indices, bins=bins, range=(0, bins))
    expected_count = len(data) / bins
    expected = np.full(bins, expected_count)

    stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    dof = bins - 1
    is_random = p_value >= alpha
    verdict = (
        f"PASS — p={p_value:.4f} ≥ α={alpha}: no significant deviation from uniformity."
        if is_random else
        f"FAIL — p={p_value:.4f} < α={alpha}: significant bias detected."
    )

    return ChiSquareResult(
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=dof,
        is_random=is_random,
        verdict=verdict,
    )


def chi_square_test_bits(bit_stream: np.ndarray, alpha: float = 0.05) -> ChiSquareResult:
    """
    Chi-square test on a binary (bit) stream.
    Expects equal numbers of 0s and 1s for a random source.

    Args:
        bit_stream: Array of 0s and 1s.
        alpha:      Significance level.

    Returns:
        ChiSquareResult with 1 degree of freedom.
    """
    n = len(bit_stream)
    count_ones = int(bit_stream.sum())
    count_zeros = n - count_ones
    observed = np.array([count_zeros, count_ones], dtype=float)
    expected = np.array([n / 2, n / 2], dtype=float)

    stat, p_value = chisquare(f_obs=observed, f_exp=expected)
    is_random = p_value >= alpha
    verdict = (
        f"PASS — p={p_value:.4f} ≥ α={alpha}: bit frequencies look balanced."
        if is_random else
        f"FAIL — p={p_value:.4f} < α={alpha}: unequal 0/1 frequencies detected."
    )

    return ChiSquareResult(
        statistic=float(stat),
        p_value=float(p_value),
        degrees_of_freedom=1,
        is_random=is_random,
        verdict=verdict,
    )


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Uniform data ===")
    uniform_data = rng.integers(0, 256, size=10000, dtype=np.uint8)
    print(chi_square_test(uniform_data))

    print("\n=== Biased data ===")
    probs = np.full(256, 1 / 256)
    probs[42] = 0.30
    probs /= probs.sum()
    biased_data = rng.choice(256, size=10000, p=probs).astype(np.uint8)
    print(chi_square_test(biased_data))
