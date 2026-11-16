import numpy as np
from scipy.linalg import eigvals as eig
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    return np.real_if_close(eig(np.array(matrix)))