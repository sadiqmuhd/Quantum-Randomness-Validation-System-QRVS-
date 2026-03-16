"""
Data Preprocessing Module
Normalizes, validates, and transforms raw entropy sequences
before they are passed to the statistical testing pipeline.
"""

import numpy as np
from typing import Tuple


# ── Normalization ────────────────────────────────────────────────────────────

def normalize_uint8(data: np.ndarray) -> np.ndarray:
    """
    Map uint8 integers [0, 255] to continuous floats in [0, 1].

    Args:
        data: Raw uint8 NumPy array.

    Returns:
        Float64 array in [0.0, 1.0].
    """
    return data.astype(np.float64) / 255.0


def normalize_uint16(data: np.ndarray) -> np.ndarray:
    """Map uint16 integers [0, 65535] to [0, 1]."""
    return data.astype(np.float64) / 65535.0


def normalize_minmax(data: np.ndarray) -> np.ndarray:
    """
    Min-max normalization to [0, 1].
    Safe against constant arrays (returns zeros).
    """
    lo, hi = data.min(), data.max()
    if hi == lo:
        return np.zeros_like(data, dtype=np.float64)
    return (data.astype(np.float64) - lo) / (hi - lo)


# ── Validation & Cleaning ────────────────────────────────────────────────────

def remove_outliers_iqr(data: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Remove statistical outliers using the IQR fence method.

    Args:
        data: Input array.
        k:    Multiplier for the IQR fence (default 3.0 for mild filtering).

    Returns:
        Array with outliers removed.
    """
    q1, q3 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (data >= lower) & (data <= upper)
    removed = len(data) - mask.sum()
    if removed:
        print(f"[Cleaner] Removed {removed} outliers (IQR fence k={k}).")
    return data[mask]


def validate_range(data: np.ndarray, lo: float = 0.0, hi: float = 255.0) -> np.ndarray:
    """
    Clamp all values to [lo, hi].  Useful after noisy simulation steps.
    """
    clamped = np.clip(data, lo, hi)
    n_clamped = int(np.sum((data < lo) | (data > hi)))
    if n_clamped:
        print(f"[Cleaner] Clamped {n_clamped} out-of-range values to [{lo}, {hi}].")
    return clamped


def drop_duplicates_streak(data: np.ndarray, max_streak: int = 10) -> np.ndarray:
    """
    Detect and report suspiciously long runs of the same value (stuck bits).
    Does not remove them — just warns.  Returns data unchanged.

    Args:
        data:       Input array.
        max_streak: Streak length considered anomalous.

    Returns:
        Original array (unchanged).
    """
    streak, longest = 1, 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 1
    if longest >= max_streak:
        print(f"[Cleaner] WARNING: Longest consecutive-value streak = {longest}. "
              f"This may indicate a stuck-bit or generator fault.")
    return data


# ── Format Conversions ───────────────────────────────────────────────────────

def to_bit_stream(data: np.ndarray) -> np.ndarray:
    """
    Unpack each uint8 value into 8 individual bits.

    Args:
        data: uint8 NumPy array.

    Returns:
        Binary NumPy array of shape (8 * len(data),) with dtype uint8.
    """
    bits = np.unpackbits(data.astype(np.uint8))
    print(f"[Cleaner] Converted {len(data)} bytes → {len(bits)}-bit stream.")
    return bits


def to_probability_distribution(data: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a normalized probability distribution (histogram) over the data.

    Args:
        data: Input array (integers or floats).
        bins: Number of histogram bins.

    Returns:
        (probabilities, bin_edges) — probabilities sum to 1.
    """
    counts, edges = np.histogram(data, bins=bins, range=(data.min(), data.max()))
    probs = counts / counts.sum()
    return probs, edges


def to_float_sequence(data: np.ndarray) -> np.ndarray:
    """Cast any integer array to float64 for use in continuous-domain tests."""
    return data.astype(np.float64)


# ── Whitening ────────────────────────────────────────────────────────────────

def whiten_histogram_equalization(data: np.ndarray) -> np.ndarray:
    """
    Whitening via histogram equalization with interpolated CDF mapping.
    Produces a more uniform spread across [0, 255] than a basic lookup,
    which resolves Chi-square failures caused by discrete CDF stepping.
    """
    data = data.astype(np.uint8)
    # Sort indices to get rank of each element
    ranks = np.argsort(np.argsort(data))   # stable double-argsort = rank
    # Map ranks linearly to [0, 255]
    n = len(data)
    whitened = (ranks * 255 / (n - 1)).astype(np.uint8)
    print(f"[Cleaner] Histogram equalization applied: {len(data)} samples whitened.")
    return whitened


# ── Convenience wrapper ──────────────────────────────────────────────────────

def preprocess(data: np.ndarray,
               remove_outliers: bool = False,
               normalize: bool = True,
               whiten: bool = False) -> np.ndarray:
    """
    Standard preprocessing pipeline applied to raw entropy data.

    Steps:
        1. Validate and clamp to uint8 range.
        2. (Optional) Remove outliers via IQR.
        3. Check for stuck-bit streaks.
        4. (Optional) Whiten via histogram equalization.
        5. (Optional) Normalize to [0, 1].

    Args:
        data:            Raw NumPy array.
        remove_outliers: Whether to apply IQR outlier removal.
        normalize:       Whether to normalize to [0.0, 1.0].
        whiten:          Whether to apply histogram equalization whitening.

    Returns:
        Processed NumPy array.
    """
    data = validate_range(data, 0, 255)
    if remove_outliers:
        data = remove_outliers_iqr(data)
    drop_duplicates_streak(data)
    if whiten:
        data = whiten_histogram_equalization(data)
    if normalize:
        data = normalize_uint8(data)
    return data


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = np.array([0, 128, 255, 300, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42], dtype=np.float32)
    cleaned = preprocess(sample, remove_outliers=False, normalize=True)
    print("Cleaned:", cleaned)