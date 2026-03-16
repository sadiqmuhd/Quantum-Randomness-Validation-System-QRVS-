"""
Shannon Entropy Estimation
Measures the average information content per symbol in a random sequence.
Maximum entropy for uint8 data is 8 bits/symbol (perfectly uniform).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ShannonEntropyResult:
    entropy_bits: float          # H(X) in bits per symbol
    max_entropy_bits: float      # Theoretical maximum (log2 of alphabet size)
    efficiency: float            # entropy / max_entropy  (1.0 = perfect)
    num_symbols: int             # Unique symbols observed
    total_samples: int
    verdict: str

    def __str__(self) -> str:
        return (
            f"Shannon Entropy\n"
            f"  H(X)       : {self.entropy_bits:.4f} bits/symbol\n"
            f"  Max H      : {self.max_entropy_bits:.4f} bits/symbol\n"
            f"  Efficiency : {self.efficiency * 100:.2f}%\n"
            f"  Symbols    : {self.num_symbols} unique / {self.total_samples} total\n"
            f"  Verdict    : {self.verdict}"
        )


def shannon_entropy(data: np.ndarray, base: float = 2.0,
                     alpha: float = 0.95) -> ShannonEntropyResult:
    """
    Compute the empirical Shannon entropy of a discrete sequence.

        H(X) = − Σ p(x) · log_base(p(x))

    Args:
        data:  Integer array (uint8 recommended).  Float arrays are quantized
               to 256 bins automatically.
        base:  Logarithm base (2 → bits, e → nats, 10 → hartleys).
        alpha: Efficiency threshold for the PASS/FAIL verdict.

    Returns:
        ShannonEntropyResult dataclass.
    """
    if np.issubdtype(data.dtype, np.floating):
        # Quantize floats in [0, 1] to 256 bins
        data = (data * 255).astype(np.uint8)

    values, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()

    # Ignore zero-probability symbols (log(0) → −∞)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs) / np.log(base))

    # Maximum possible entropy for the observed alphabet
    max_entropy = np.log(256) / np.log(base)  # log2(256) = 8 bits
    efficiency = entropy / max_entropy if max_entropy > 0 else 0.0

    is_high = efficiency >= alpha
    verdict = (
        f"GOOD — efficiency {efficiency * 100:.2f}% ≥ {alpha * 100:.0f}%: "
        f"high-quality entropy source."
        if is_high else
        f"WEAK — efficiency {efficiency * 100:.2f}% < {alpha * 100:.0f}%: "
        f"entropy is below expectations for a strong random source."
    )

    return ShannonEntropyResult(
        entropy_bits=float(entropy),
        max_entropy_bits=float(max_entropy),
        efficiency=float(efficiency),
        num_symbols=len(values),
        total_samples=len(data),
        verdict=verdict,
    )


def block_entropy(data: np.ndarray, block_size: int = 8,
                   base: float = 2.0) -> np.ndarray:
    """
    Compute Shannon entropy over rolling blocks of `block_size` samples.
    Useful for detecting local non-stationarity (entropy changes over time).

    Args:
        data:       Integer array.
        block_size: Number of samples per block.
        base:       Logarithm base.

    Returns:
        Array of per-block entropy values.
    """
    n_blocks = len(data) // block_size
    entropies = []

    for i in range(n_blocks):
        block = data[i * block_size:(i + 1) * block_size]
        result = shannon_entropy(block, base=base)
        entropies.append(result.entropy_bits)

    return np.array(entropies)


def conditional_entropy(x: np.ndarray, y: np.ndarray,
                         base: float = 2.0) -> float:
    """
    Compute H(X | Y) — the conditional Shannon entropy of X given Y.
    Low H(X|Y) indicates X is predictable from Y (i.e. strong correlation).

    Args:
        x, y: Equal-length integer arrays (uint8 recommended).
        base: Logarithm base.

    Returns:
        Conditional entropy in the chosen unit.
    """
    assert len(x) == len(y), "x and y must have the same length"
    n = len(x)

    # Joint probability table
    joint = {}
    for xi, yi in zip(x, y):
        key = (int(xi), int(yi))
        joint[key] = joint.get(key, 0) + 1

    y_counts: dict = {}
    for (xi, yi), cnt in joint.items():
        y_counts[yi] = y_counts.get(yi, 0) + cnt

    h_x_given_y = 0.0
    for (xi, yi), cnt in joint.items():
        p_xy = cnt / n
        p_y  = y_counts[yi] / n
        p_x_given_y = p_xy / p_y
        h_x_given_y -= p_xy * np.log(p_x_given_y) / np.log(base)

    return h_x_given_y


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Uniform uint8 ===")
    uniform = rng.integers(0, 256, size=10000, dtype=np.uint8)
    print(shannon_entropy(uniform))

    print("\n=== Single value (zero entropy) ===")
    constant = np.full(1000, 42, dtype=np.uint8)
    print(shannon_entropy(constant))

    print("\n=== Block entropy (first 5 blocks) ===")
    be = block_entropy(uniform, block_size=256)
    print("Block entropies:", be[:5].round(3))
