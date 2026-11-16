import numpy as np


def transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
) -> list[list[int | float]] | int:
  try:
    A_np = np.array(A, dtype=float)
    T_np = np.array(T, dtype=float)
    S_np = np.array(S, dtype=float)
 
    if T_np.shape[0] != T_np.shape[1] or S_np.shape[0] != S_np.shape[1]:
        return -1
    if np.isclose(np.linalg.det(T_np), 0) or np.isclose(np.linalg.det(S_np), 0):
        return -1

    result = np.linalg.solve(T_np, A_np) @ S_np

    return result.tolist()
  except (np.linalg.LinAlgError, ValueError):
    return -1
