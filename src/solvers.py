#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/solvers.py
================================================================================
PPE-MNLS (lsqr and DCT) and RPR-ODI (OS-MODI) solvers.

LINEAGE / WHY TWO PPE SOLVERS COEXIST HERE
-------------------------------------------
This project's solver choice evolved over the course of the internship:

  - Section 5 (deterministic results) was produced with `solve_PPE_MNLS`,
    a sparse `scipy.sparse.linalg.lsqr` solve of the discrete Neumann
    Laplacian system L_N p = b_N. This was the first solver implemented.

  - Starting with Section 6 (gradient-noise sensitivity) and used through
    Sections 7-8, the project switched to `solve_poisson_DCT`: an exact,
    O(N^2 log N) solver that exploits the fact that L_N is diagonalized by
    the DCT-II (Neumann eigenbasis). It is verified in Section 8 to
    reproduce the lsqr solution to ~1e-15 while being 1000-6000x faster,
    and it avoids an lsqr convergence artifact at N=128 documented in the
    same section.

`lsqr` is NOT deprecated dead code: it is the solver behind every Section 5
figure and is used again, deliberately, in Section 8's solver-comparison
benchmark. Both are kept here, side by side, without one being marked as
the "correct" one -- which one is authoritative depends on which section
of the report you are reproducing (see experiments/ and the top-level
README for the section -> script mapping).
================================================================================
"""

import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.fft import dctn, idctn

try:
    from osmodi import solve_cpu, CPU_AVAILABLE
except ImportError:
    CPU_AVAILABLE = False
    print("[WARNING] Package OS-MODI (osmodi) non détecté. "
          "solve_OS_MODI ne pourra pas être appelé.")

try:
    from scipy.sparse.linalg import lsqr as _lsqr
except ImportError:
    _lsqr = None


# =============================================================================
# Shared system assembly
# =============================================================================

def build_LN_bN(dPdx, dPdz, h):
    """Assembles the discrete Neumann Laplacian L_N (5-point reflecting
    stencil, with flux treatment at the boundaries) and the RHS b_N from
    a pair of pressure-gradient fields dPdx, dPdz on an N x N grid.

    Identical in both legacy scripts (section5rapportfinal.py and
    finalpipeline.py) -- consolidated here as the single source of truth.

    Returns
    -------
    LN : scipy.sparse.csr_matrix, shape (N*N, N*N)
    bN : np.ndarray, shape (N*N,)
    """
    N = dPdx.shape[0]
    n = N * N

    def idx(i, j):
        return i * N + j

    LN = lil_matrix((n, n), dtype=np.float64)
    bN = np.zeros(n, dtype=np.float64)

    for i in range(N):
        for j in range(N):
            k = idx(i, j)
            neighbors = []
            if i > 0:
                neighbors.append(idx(i - 1, j))
            if i < N - 1:
                neighbors.append(idx(i + 1, j))
            if j > 0:
                neighbors.append(idx(i, j - 1))
            if j < N - 1:
                neighbors.append(idx(i, j + 1))

            nb = len(neighbors)
            LN[k, k] = -nb / h**2
            for kk in neighbors:
                LN[k, kk] = 1.0 / h**2

            if 0 < i < N - 1 and 0 < j < N - 1:
                div_x = (dPdx[i + 1, j] - dPdx[i - 1, j]) / (2 * h)
                div_z = (dPdz[i, j + 1] - dPdz[i, j - 1]) / (2 * h)
                bN[k] = div_x + div_z
            else:
                if i == 0:
                    flux_x = (dPdx[0, j] + dPdx[1, j]) / (2 * h)
                elif i == N - 1:
                    flux_x = -(dPdx[N - 1, j] + dPdx[N - 2, j]) / (2 * h)
                else:
                    flux_x = (dPdx[i + 1, j] - dPdx[i - 1, j]) / (2 * h)

                if j == 0:
                    flux_z = (dPdz[i, 0] + dPdz[i, 1]) / (2 * h)
                elif j == N - 1:
                    flux_z = -(dPdz[i, N - 1] + dPdz[i, N - 2]) / (2 * h)
                else:
                    flux_z = (dPdz[i, j + 1] - dPdz[i, j - 1]) / (2 * h)

                bN[k] = flux_x + flux_z

    return LN.tocsr(), bN


# =============================================================================
# PPE-MNLS -- sparse lsqr solver (Section 5 solver; also used in Section 8
# for the lsqr-vs-DCT-vs-RPR-ODI runtime/accuracy comparison)
# =============================================================================

def solve_PPE_MNLS(LN, bN, N, return_runtime=False, atol=1e-7, btol=1e-7):
    """Minimum-norm least-squares solve of L_N p = b_N via
    scipy.sparse.linalg.lsqr.

    Parameters
    ----------
    return_runtime : if True, returns (p_centered, runtime_seconds), matching
        the calling convention used in section5rapportfinal.py. If False
        (default), returns only p_centered, matching finalpipeline.py.
    """
    if _lsqr is None:
        raise ImportError("scipy.sparse.linalg.lsqr is unavailable.")
    t0 = time.time()
    res = _lsqr(LN, bN, atol=atol, btol=btol)
    runtime = time.time() - t0
    p_flat = res[0]
    p_centered = p_flat.reshape(N, N) - p_flat.mean()
    if return_runtime:
        return p_centered, runtime
    return p_centered


# =============================================================================
# PPE-MNLS -- fast exact DCT solver (Sections 6-8 default)
# =============================================================================

def neumann_eigenvalues(N, h):
    """Eigenvalues of L_N in the DCT-II (Neumann) eigenbasis."""
    m = np.arange(N)
    lam_1d = -2.0 / h**2 * (1.0 - np.cos(np.pi * m / N))
    Lx, Lz = np.meshgrid(lam_1d, lam_1d, indexing='ij')
    return Lx + Lz


def solve_poisson_DCT(bN_2d, h):
    """Exact, fast solve of L_N p = b_N via forward DCT -> elementwise
    division by the Neumann eigenvalues -> inverse DCT. O(N^2 log N).
    Identical to solve_PPE_MNLS to ~1e-15 (verified in Section 8)."""
    N = bN_2d.shape[0]
    eig = neumann_eigenvalues(N, h)
    B_hat = dctn(bN_2d, type=2, norm='ortho')
    P_hat = np.zeros_like(B_hat)
    mask = np.abs(eig) > 1e-12
    P_hat[mask] = B_hat[mask] / eig[mask]
    P_hat[0, 0] = 0.0  # zero-mean gauge (null space of L_N)
    p = idctn(P_hat, type=2, norm='ortho')
    return p - p.mean()


def solve_poisson_DCT_tikhonov(bN_2d, h, gamma):
    """Same solver with Tikhonov regularization (||p||^2 penalty).
    gamma=0 reproduces solve_poisson_DCT exactly. Used in Section 8."""
    N = bN_2d.shape[0]
    eig = neumann_eigenvalues(N, h)
    B_hat = dctn(bN_2d, type=2, norm='ortho')
    P_hat = np.zeros_like(B_hat)
    mask = np.ones_like(eig, dtype=bool)
    mask[0, 0] = False
    P_hat[mask] = (eig[mask] * B_hat[mask]) / (eig[mask]**2 + gamma)
    P_hat[0, 0] = 0.0
    p = idctn(P_hat, type=2, norm='ortho')
    return p - p.mean()


# =============================================================================
# RPR-ODI (OS-MODI reference implementation)
# =============================================================================

def solve_OS_MODI(dPdx, dPdz, h, tol=1e-6, max_iter=1500, return_runtime=False):
    """Resolution via the reference OS-MODI (C++/pressure-osmosis) package
    (Zigunov & Charonko). Wraps the 2D fields into the 3D interface expected
    by solve_cpu with a single, empty out-of-plane layer.
    """
    if not CPU_AVAILABLE:
        raise ImportError("osmodi.solve_cpu is unavailable (CPU_AVAILABLE=False).")

    Sx_3d = np.array(dPdx[:, np.newaxis, :], dtype=np.float64)
    Sy_3d = np.zeros_like(Sx_3d, dtype=np.float64)
    Sz_3d = np.array(dPdz[:, np.newaxis, :], dtype=np.float64)
    delta = np.array([h, h, h], dtype=np.float64)
    options = {'Tolerance': tol, 'MaxIterations': max_iter}

    t0 = time.time()
    solver_output = solve_cpu(Sx_3d, Sy_3d, Sz_3d, delta, options)
    runtime = time.time() - t0

    if isinstance(solver_output, tuple):
        P_official_3d = solver_output[0]
        convergence_history = solver_output[1]
        if isinstance(convergence_history, tuple):
            convergence_history = convergence_history[0]
        total_iters = int(convergence_history[-1, 0])
        final_res = convergence_history[-1, 1]
        print(f"-> OS-MODI converged in {total_iters} iterations "
              f"(final residual: {final_res:.4e})")
    else:
        P_official_3d = solver_output

    P_modi = P_official_3d[:, 0, :]
    p_centered = P_modi - P_modi.mean()
    if return_runtime:
        return p_centered, runtime
    return p_centered
