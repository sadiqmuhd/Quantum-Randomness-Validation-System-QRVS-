"""
Min-Entropy Estimation
Measures the worst-case unpredictability of a random source.
Used in cryptographic contexts (NIST SP 800-90B).

Min-entropy = − log2(max_probability)

A perfect uint8 source has min-entropy = 8 bits/symbol.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class MinEntropyResult:
    min_entropy_bits: float      # Hmin = −log2(p_max)
    max_entropy_bits: float      # Theoretical max (log2 of alphabet size)
    p_max: float                 # Probability of the most common symbol
    most_common_symbol: int      # The dominant symbol
    efficiency: float            # min_entropy / max_entropy
    verdict: str

    def __str__(self) -> str:
        return (
            f"Min-Entropy\n"
            f"  Hmin             : {self.min_entropy_bits:.4f} bits/symbol\n"
            f"  Max H            : {self.max_entropy_bits:.4f} bits/symbol\n"
            f"  p_max            : {self.p_max:.6f} (symbol={self.most_common_symbol})\n"
            f"  Efficiency       : {self.efficiency * 100:.2f}%\n"
            f"  Verdict          : {self.verdict}"
        )


def min_entropy(data: np.ndarray, alpha: float = 0.90) -> MinEntropyResult:
    """
    Estimate min-entropy from the empirical symbol frequencies.

    A strong random source should have min-entropy close to log2(alphabet_size).

    Args:
        data:  Integer array (uint8 recommended).  Float arrays in [0, 1]
               are quantized to 256 bins.
        alpha: Efficiency threshold below which the source is flagged WEAK.

    Returns:
        MinEntropyResult dataclass.
    """
    if np.issubdtype(data.dtype, np.floating):
        data = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)

    values, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()

    idx_max = int(np.argmax(probs))
    p_max   = float(probs[idx_max])
    dominant_symbol = int(values[idx_max])

    hmin = -np.log2(p_max)
    max_entropy = np.log2(256)          # 8 bits for uint8 alphabet
    efficiency  = hmin / max_entropy

    is_good = efficiency >= alpha
    verdict = (
        f"GOOD — efficiency {efficiency * 100:.2f}% ≥ {alpha * 100:.0f}%: "
        f"dominant symbol probability ({p_max:.4f}) is acceptably low."
        if is_good else
        f"WEAK — efficiency {efficiency * 100:.2f}% < {alpha * 100:.0f}%: "
        f"symbol {dominant_symbol} appears too frequently (p={p_max:.4f}). "
        f"Entropy quality is insufficient for cryptographic use."
    )

    return MinEntropyResult(
        min_entropy_bits=float(hmin),
        max_entropy_bits=float(max_entropy),
        p_max=p_max,
        most_common_symbol=dominant_symbol,
        efficiency=float(efficiency),
        verdict=verdict,
    )


def collision_entropy(data: np.ndarray, base: float = 2.0) -> float:
    """
    Compute the Rényi collision entropy H₂:

        H₂(X) = −log( Σ p(x)² )

    This is the order-2 Rényi entropy and measures how likely two
    independent draws from the distribution are to collide.

    Args:
        data: Integer array.
        base: Logarithm base.

    Returns:
        Collision entropy value in the chosen unit.
    """
    if np.issubdtype(data.dtype, np.floating):
        data = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)

    _, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.log(np.sum(probs ** 2)) / np.log(base))


def guessing_entropy(data: np.ndarray) -> float:
    """
    Estimate guessing entropy G(X): the expected number of guesses required
    to identify the value of a random draw when symbols are tried
    most-likely-first.

        G(X) = Σ i · p_{(i)}

    where p_{(1)} ≥ p_{(2)} ≥ … (sorted in descending order).

    Args:
        data: Integer array.

    Returns:
        Expected number of guesses.
    """
    _, counts = np.unique(data, return_counts=True)
    probs = np.sort(counts / counts.sum())[::-1]   # descending
    ranks = np.arange(1, len(probs) + 1)
    return float(np.dot(ranks, probs))


def nist_iid_min_entropy(data: np.ndarray) -> float:
    """
    Simplified NIST SP 800-90B IID min-entropy estimate.
    Uses the most common value estimator (Appendix C.1).

    Returns min-entropy in bits per symbol.
    """
    _, counts = np.unique(data, return_counts=True)
    n = len(data)
    p_hat = counts.max() / n
    # Apply NIST upper-bound correction
    p_upper = min(1.0, p_hat + 2.576 * np.sqrt(p_hat * (1 - p_hat) / n))
    return float(-np.log2(p_upper))


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Uniform uint8 ===")
    uniform = rng.integers(0, 256, size=10000, dtype=np.uint8)
    print(min_entropy(uniform))
    print(f"  Collision entropy : {collision_entropy(uniform):.4f} bits")
    print(f"  Guessing entropy  : {guessing_entropy(uniform):.2f} guesses")
    print(f"  NIST IID estimate : {nist_iid_min_entropy(uniform):.4f} bits")

    print("\n=== Biased source ===")
    probs_b = np.full(256, 1 / 256)
    probs_b[42] = 0.40
    probs_b /= probs_b.sum()
    biased = rng.choice(256, size=10000, p=probs_b).astype(np.uint8)
    print(min_entropy(biased))
