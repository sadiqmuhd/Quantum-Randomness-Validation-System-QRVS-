"""
Pseudo Random Number Generator
Provides baseline random sequences using NumPy's PRNG engines.
Used as a comparison baseline against quantum randomness sources.
"""

import numpy as np
from pathlib import Path


DATA_DIR = Path.cwd() / "data" / "raw"


def generate_uniform(n: int = 1000, seed: int | None = None) -> np.ndarray:
    """
    Generate uniformly distributed random integers in [0, 255] (uint8 range)
    to match the ANU QRNG output format.

    Args:
        n:    Number of values to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        NumPy array of uint8 integers.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=n, dtype=np.uint8)
    print(f"[Pseudo RNG] Generated {n} uniform uint8 values.")
    return data


def generate_continuous_uniform(n: int = 1000, seed: int | None = None) -> np.ndarray:
    """
    Generate continuous uniform floats in [0, 1).

    Args:
        n:    Number of values.
        seed: Optional random seed.

    Returns:
        NumPy array of float64 in [0, 1).
    """
    rng = np.random.default_rng(seed)
    data = rng.random(size=n)
    print(f"[Pseudo RNG] Generated {n} continuous uniform values.")
    return data


def generate_biased(n: int = 1000, bias_value: int = 42, bias_weight: float = 0.3,
                    seed: int | None = None) -> np.ndarray:
    """
    Generate a biased sequence where one value appears more frequently.
    Useful for testing that statistical tests correctly flag non-uniform data.

    Args:
        n:            Number of values.
        bias_value:   The uint8 value that is over-represented.
        bias_weight:  Probability of drawing the biased value (0 < bias_weight < 1).
        seed:         Optional random seed.

    Returns:
        NumPy array of uint8 integers with introduced bias.
    """
    rng = np.random.default_rng(seed)
    # Build a non-uniform probability vector
    probs = np.full(256, (1.0 - bias_weight) / 255)
    probs[bias_value] = bias_weight
    data = rng.choice(np.arange(256, dtype=np.uint8), size=n, p=probs)
    print(f"[Pseudo RNG] Generated {n} biased values (bias_value={bias_value}, weight={bias_weight}).")
    return data


def generate_lcg(n: int = 1000, seed: int = 12345) -> np.ndarray:
    """
    Linear Congruential Generator — a deliberately weak PRNG.
    Demonstrates high autocorrelation and poor randomness quality.

    Parameters chosen to give a visible pattern:
        x_{n+1} = (a * x_n + c) mod m

    Args:
        n:    Number of values.
        seed: Starting value.

    Returns:
        NumPy array of uint8-range integers (values mod 256).
    """
    a, c, m = 1664525, 1013904223, 2**32
    values = np.empty(n, dtype=np.uint64)
    x = seed
    for i in range(n):
        x = (a * x + c) % m
        values[i] = x
    data = (values % 256).astype(np.uint8)
    print(f"[Pseudo RNG] Generated {n} LCG values (seed={seed}).")
    return data


def save_pseudo_data(data: np.ndarray, filename: str = "pseudo_rng_data.npy") -> Path:
    """Save generated data to data/raw/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    np.save(filepath, data)
    print(f"[Pseudo RNG] Saved to {filepath}")
    return filepath


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uniform = generate_uniform(n=10, seed=0)
    biased  = generate_biased(n=10, seed=0)
    lcg     = generate_lcg(n=10)
    print("Uniform:", uniform)
    print("Biased :", biased)
    print("LCG    :", lcg)
