import numpy as np


def solve_jacobi(
    A: list[list[float]], b: list[float], n: int
) -> list[float]:
  A_np = np.array(A, dtype=np.float64)
  b_np = np.array(b, dtype=np.float64)

  D = np.diag(A_np)
  R = A_np - np.diagflat(D)
  x = np.zeros_like(b_np, dtype=np.float64)

  for _ in range(n):
    x = (b_np - R @ x) / D
  return np.round(x, 4).tolist()
