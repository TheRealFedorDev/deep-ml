import numpy as np

def bucketize(v: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Return bucket indices for v given sorted boundaries."""
    v = np.asarray(v)
    boundaries = np.asarray(boundaries)
    return (v[:, None] >= boundaries).sum(axis=1)