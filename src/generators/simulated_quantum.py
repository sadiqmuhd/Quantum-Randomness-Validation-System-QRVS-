"""
Simulated Quantum Noise Generator
Approximates quantum-like randomness using physical noise models
(Gaussian, Poisson, phase noise) without requiring real hardware.
"""

import numpy as np
from pathlib import Path


DATA_DIR = Path.cwd() / "data" / "raw"


def generate_gaussian_noise(n: int = 1000, mean: float = 127.5, std: float = 40.0,
                             seed: int | None = None) -> np.ndarray:
    """
    Simulate photon-shot noise or thermal noise using a Gaussian distribution,
    then clamp and quantize to uint8.

    Args:
        n:    Number of samples.
        mean: Centre of the distribution.
        std:  Standard deviation (controls spread across 0–255).
        seed: Optional random seed.

    Returns:
        NumPy uint8 array.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(loc=mean, scale=std, size=n)
    data = np.clip(raw, 0, 255).astype(np.uint8)
    print(f"[Sim Quantum] Generated {n} Gaussian noise samples (μ={mean}, σ={std}).")
    return data


def generate_poisson_noise(n: int = 1000, lam: float = 127.5,
                            seed: int | None = None) -> np.ndarray:
    """
    Simulate photon-counting noise using a Poisson distribution.
    Mimics quantum shot noise in photodetectors.

    Args:
        n:   Number of samples.
        lam: Expected photon count (λ). Values are clamped to [0, 255].
        seed: Optional random seed.

    Returns:
        NumPy uint8 array.
    """
    rng = np.random.default_rng(seed)
    raw = rng.poisson(lam=lam, size=n)
    data = np.clip(raw, 0, 255).astype(np.uint8)
    print(f"[Sim Quantum] Generated {n} Poisson noise samples (λ={lam}).")
    return data


def generate_phase_noise(n: int = 1000, seed: int | None = None) -> np.ndarray:
    """
    Simulate quantum phase noise by sampling random phases on the unit circle
    and mapping to uint8. Models laser phase diffusion.

    Args:
        n:    Number of samples.
        seed: Optional random seed.

    Returns:
        NumPy uint8 array derived from random phase angles.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, size=n)
    # Map [-1, 1] sine range → [0, 255]
    normalized = (np.sin(phases) + 1.0) / 2.0
    data = (normalized * 255).astype(np.uint8)
    print(f"[Sim Quantum] Generated {n} phase-noise samples.")
    return data


def generate_vacuum_fluctuation(n: int = 1000, seed: int | None = None) -> np.ndarray:
    """
    Simulate quantum vacuum fluctuations by superposing two independent
    Gaussian quadratures (X and P) and extracting the magnitude, quantized
    to uint8.  This loosely models homodyne detection of a vacuum state.

    Args:
        n:    Number of samples.
        seed: Optional random seed.

    Returns:
        NumPy uint8 array.
    """
    rng = np.random.default_rng(seed)
    x_quad = rng.normal(0, 1, size=n)
    p_quad = rng.normal(0, 1, size=n)
    amplitude = np.sqrt(x_quad**2 + p_quad**2)          # Rayleigh-distributed
    # Rayleigh mode ≈ 1, scale to fill uint8 range
    scale_factor = 255 / (amplitude.max() + 1e-9)
    data = np.clip(amplitude * scale_factor, 0, 255).astype(np.uint8)
    print(f"[Sim Quantum] Generated {n} vacuum-fluctuation samples.")
    return data


def save_simulated_data(data: np.ndarray, filename: str = "simulated_quantum_data.npy") -> Path:
    """Persist simulated data to data/raw/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    np.save(filepath, data)
    print(f"[Sim Quantum] Saved to {filepath}")
    return filepath


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Gaussian :", generate_gaussian_noise(n=8, seed=0))
    print("Poisson  :", generate_poisson_noise(n=8, seed=0))
    print("Phase    :", generate_phase_noise(n=8, seed=0))
    print("Vacuum   :", generate_vacuum_fluctuation(n=8, seed=0))
