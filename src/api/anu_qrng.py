"""
ANU Quantum Random Number Generator API Client
Fetches true quantum random numbers from the Australian National University QRNG API.
Source: https://qrng.anu.edu.au
"""

import requests
import numpy as np
import json
import os
import time
from pathlib import Path


ANU_API_URL = "https://qrng.anu.edu.au/API/jsonI.php"
DATA_DIR = Path.cwd() / "data" / "raw"


def fetch_qrng_data(length: int = 1000, data_type: str = "uint8", retries: int = 3) -> np.ndarray:
    """
    Fetch quantum random numbers from the ANU QRNG API.

    Args:
        length:    Number of random values to fetch (max 1024 per request).
        data_type: One of 'uint8' (0–255) or 'uint16' (0–65535).
        retries:   Number of retry attempts on failure.

    Returns:
        NumPy array of random integers.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    params = {
        "length": min(length, 1024),
        "type": data_type,
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"[ANU QRNG] Fetching {length} {data_type} values (attempt {attempt}/{retries})...")
            response = requests.get(ANU_API_URL, params=params, timeout=15)
            response.raise_for_status()

            payload = response.json()
            if not payload.get("success"):
                raise ValueError(f"API returned failure: {payload}")

            data = np.array(payload["data"], dtype=np.uint16 if data_type == "uint16" else np.uint8)
            print(f"[ANU QRNG] Successfully fetched {len(data)} values.")
            return data

        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"[ANU QRNG] Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # exponential back-off

    raise RuntimeError("[ANU QRNG] All retry attempts failed. Check your internet connection or the API status.")


def fetch_large_qrng_dataset(total: int = 5000, data_type: str = "uint8") -> np.ndarray:
    """
    Fetch more than 1024 values by batching multiple API requests.

    Args:
        total:     Total number of random values required.
        data_type: 'uint8' or 'uint16'.

    Returns:
        Concatenated NumPy array of the requested length.
    """
    batch_size = 1024
    chunks = []
    remaining = total

    while remaining > 0:
        fetch_count = min(batch_size, remaining)
        chunk = fetch_qrng_data(length=fetch_count, data_type=data_type)
        chunks.append(chunk)
        remaining -= fetch_count
        if remaining > 0:
            time.sleep(0.5)  # be polite to the API

    result = np.concatenate(chunks)
    print(f"[ANU QRNG] Total dataset assembled: {len(result)} values.")
    return result


def save_qrng_data(data: np.ndarray, filename: str = "qrng_data.npy") -> Path:
    """
    Persist the fetched QRNG data to disk.

    Args:
        data:     NumPy array to save.
        filename: Output filename inside data/raw/.

    Returns:
        Absolute path to the saved file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename
    np.save(filepath, data)
    print(f"[ANU QRNG] Data saved to {filepath}")
    return filepath


def load_qrng_data(filename: str = "qrng_data.npy") -> np.ndarray:
    """
    Load previously saved QRNG data from disk.

    Args:
        filename: File inside data/raw/ to load.

    Returns:
        NumPy array of the saved data.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    filepath = DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"[ANU QRNG] File not found: {filepath}")
    data = np.load(filepath)
    print(f"[ANU QRNG] Loaded {len(data)} values from {filepath}")
    return data


def get_or_fetch_qrng(total: int = 1000, filename: str = "qrng_data.npy", force_refresh: bool = False) -> np.ndarray:
    """
    Return cached QRNG data if available, otherwise fetch from API and cache.

    Args:
        total:         Number of values to fetch if cache miss.
        filename:      Cache filename in data/raw/.
        force_refresh: Re-fetch even if cache exists.

    Returns:
        NumPy array of quantum random numbers.
    """
    filepath = DATA_DIR / filename
    if not force_refresh and filepath.exists():
        print("[ANU QRNG] Using cached data.")
        return load_qrng_data(filename)

    data = fetch_large_qrng_dataset(total=total)
    save_qrng_data(data, filename)
    return data


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = fetch_qrng_data(length=10)
    print("Sample QRNG values:", sample)
