import numpy as np


def solve_linear_ode(R, V, P_soft, B, lambda_ode: float = 1.0):
    """ALS for W with ODE ṗ = A r − b p, p = W r. b is mean(B): B lives in
    protein space, not RNA space.
    """
    b = float(np.mean(B))
    W = np.linalg.lstsq(R, P_soft, rcond=None)[0].T
    VbR = V + b * R
    for _ in range(50):
        lhs = (W @ VbR.T).T
        A = np.linalg.lstsq(R, lhs, rcond=None)[0].T
        AR = (A @ R.T).T
        lhs_full = np.vstack([R, np.sqrt(lambda_ode) * VbR])
        rhs_full = np.vstack([P_soft, np.sqrt(lambda_ode) * AR])
        W = np.linalg.lstsq(lhs_full, rhs_full, rcond=None)[0].T

    return W, A
