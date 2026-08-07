#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/metrics.py
================================================================================
Shared error/correlation metrics and finite-difference helpers used across
Sections 5-8 (gradient reconstruction from noisy velocity, non-periodic
Laplacian, R / relative-RMS-error metric pair).
================================================================================
"""

import numpy as np


def R_and_err(p, p_ref):
    """Spatial correlation R and relative RMS error against a reference
    field, the metric pair used throughout the report (Section 4.4)."""
    R = np.corrcoef(p_ref.flatten(), p.flatten())[0, 1]
    eps_rel = np.sqrt(np.mean((p - p_ref) ** 2)) / np.sqrt(np.mean(p_ref ** 2))
    return R, eps_rel


def laplacian_nonperiodic(f, h):
    """2D Laplacian with second-order one-sided schemes at the boundaries
    (non-periodic, matches JHTDB's finite plane extraction)."""
    lap = np.zeros_like(f)
    # Interior: standard second-order centered 3-point stencil.
    lap[1:-1, :] += (f[2:, :] - 2 * f[1:-1, :] + f[:-2, :]) / h**2
    # Boundaries: no neighbor on one side, so a centered stencil isn't
    # available. Use a one-sided, second-order-accurate 4-point stencil
    # instead (coefficients 2, -5, 4, -1), applied along each axis
    # independently at its own edge.
    lap[0, :]    += (2 * f[0, :] - 5 * f[1, :] + 4 * f[2, :] - f[3, :]) / h**2
    lap[-1, :]   += (2 * f[-1, :] - 5 * f[-2, :] + 4 * f[-3, :] - f[-4, :]) / h**2
    lap[:, 1:-1] += (f[:, 2:] - 2 * f[:, 1:-1] + f[:, :-2]) / h**2
    lap[:, 0]    += (2 * f[:, 0] - 5 * f[:, 1] + 4 * f[:, 2] - f[:, 3]) / h**2
    lap[:, -1]   += (2 * f[:, -1] - 5 * f[:, -2] + 4 * f[:, -3] - f[:, -4]) / h**2
    return lap


def gradients_from_velocity(u_x, u_y, u_z, u_x_prev, u_z_prev, u_x_next, u_z_next,
                             h, dt, nu, du_x_dy, du_z_dy):
    """Reconstructs dPdx, dPdz from (possibly noisy) velocity fields via the
    Navier-Stokes momentum equation. Used in Section 7 to propagate velocity
    noise through the unsteady acceleration term du/dt = (u^{n+1}-u^{n-1})/(2dt)
    before it reaches the pressure solvers.
    """
    du_x_dt = (u_x_next - u_x_prev) / (2 * dt)
    du_z_dt = (u_z_next - u_z_prev) / (2 * dt)

    du_x_dx = np.gradient(u_x, h, axis=0)
    du_x_dz = np.gradient(u_x, h, axis=1)
    du_z_dx = np.gradient(u_z, h, axis=0)
    du_z_dz = np.gradient(u_z, h, axis=1)

    lap_ux = laplacian_nonperiodic(u_x, h)
    lap_uz = laplacian_nonperiodic(u_z, h)

    dPdx = -(du_x_dt + u_x * du_x_dx + u_y * du_x_dy + u_z * du_x_dz) + nu * lap_ux
    dPdz = -(du_z_dt + u_x * du_z_dx + u_y * du_z_dy + u_z * du_z_dz) + nu * lap_uz
    return dPdx, dPdz
