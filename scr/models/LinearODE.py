import numpy as np


def solve_linear_ode(R, V, P_soft, B, lambda_ode: float = 1.0):
    """
    Solve for linear map W: R^D_r → R^D_p via alternating least squares.

    ODE: ṗ = A·r - b·p,  p = W·r  →  W·v = A·r - b·W·r
    → W·(v + b·r) = A·r   (scalar b = mean degradation rate across proteins)

    Parameters
    ----------
    R       : (n, D_r)  RNA representations
    V       : (n, D_r)  RNA velocity vectors
    P_soft  : (n, D_p)  soft-matched protein targets (from Sinkhorn coupling)
    B       : (D_p,)    per-protein degradation rates; mean used as scalar b
    lambda_ode : float  weight of ODE regularization

    Returns
    -------
    W : (D_p, D_r)  linear translation map
    A : (D_p, D_r)  translation efficiency matrix
    """
    b = float(np.mean(B))     # scalar: B lives in protein space, not RNA space

    # Initialize W via ordinary least squares: P_soft ≈ W R^T
    W = np.linalg.lstsq(R, P_soft, rcond=None)[0].T   # (D_p, D_r)

    VbR = V + b * R           # (n, D_r)  — fixed for the whole ALS run

    for _ in range(50):
        # Step 1: update A given W
        # W·(v + b·r) = A·r  →  solve R @ A^T ≈ (V+b·R) @ W^T
        lhs = (W @ VbR.T).T                                          # (n, D_p)
        A = np.linalg.lstsq(R, lhs, rcond=None)[0].T                # (D_p, D_r)

        # Step 2: update W given A
        # min_W ||P_soft - W·R||^2 + λ ||W·(V+b·R) - A·R||^2
        AR = (A @ R.T).T                                             # (n, D_p)
        lhs_full = np.vstack([R,           np.sqrt(lambda_ode) * VbR])   # (2n, D_r)
        rhs_full = np.vstack([P_soft,      np.sqrt(lambda_ode) * AR ])   # (2n, D_p)
        W = np.linalg.lstsq(lhs_full, rhs_full, rcond=None)[0].T    # (D_p, D_r)

    return W, A
