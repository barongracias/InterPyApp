"""Synthetic data generators for 5D->1 interpolation experiments."""

# imports
import os
import math
import pickle
from datetime import datetime, timezone
from typing import Tuple

import numpy as np


def synthetic_5d(n: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic 5D inputs and a smooth target surface with noise.

    Args:
        n (int): Number of samples to generate.
        seed (int | None): Optional RNG seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix ``X`` of shape ``(n, 5)`` and
        target vector ``y`` of shape ``(n, 1)``, both ``float32``.
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 5))
    x1, x2, x3, x4, x5 = X.T

    y = (
        np.sin(2 * math.pi * x1) * np.cos(2 * math.pi * x2)
        + 0.3 * np.exp(-((x3 - 0.5) ** 2 + (x4 - 0.5) ** 2) / 0.02)
        + 0.5 * x5**2
        - 0.2 * x1 * x4
    )
    y += rng.normal(0, 0.01, size=n)

    return X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)


def synthetic_5d_pickle(path: str, n: int, seed: int | None = None) -> str:
    """
    Generate and persist a synthetic 5D dataset to pickle with metadata.

    Args:
        path (str): Target filepath (including ``.pkl`` name) to write.
        n (int): Number of samples to generate.
        seed (int | None): Optional RNG seed for reproducibility.

    Returns:
        str: The absolute or relative path written.

    Raises:
        OSError: If the file cannot be written to the requested location.
    """
    X, y = synthetic_5d(n, seed=seed)
    metadata = {
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "seed": seed,
        "feature_names": [f"x{i+1}" for i in range(X.shape[1])],
        "targetname": "y",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = {"X": X, "y": y, "metadata": metadata}

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path
