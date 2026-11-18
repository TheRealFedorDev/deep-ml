import numpy as np

def kron_matvec(A, B, C):
    M = B @ C @ A.T
    return M.reshape(-1, order="F")

def kron_solve(A, B, y):
    n = A.shape[0]
    q = B.shape[0]
    Y = np.asarray(y, dtype=float).reshape((q, n), order="F")
    Z = np.linalg.solve(B, Y)
    X = np.linalg.solve(A, Z.T).T
    return X.reshape(-1, order="F")

def bow_logits(S, W, V):
    return np.einsum("ntk,kd,dc->nc", S, W, V, optimize=True)

