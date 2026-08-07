#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/jhtdb_io.py
================================================================================
JHTDB (isotropic1024coarse) connection and field-extraction utilities.

The JHTDB auth token is read from the environment variable JHTDB_AUTH_TOKEN
(see .env.example at the repo root) rather than hardcoded, so this file is
safe to commit to a public or shared repo.
================================================================================
"""

import os
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads JHTDB_AUTH_TOKEN from a local .env file, if present
except ImportError:
    pass

try:
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData
    JHTDB_AVAILABLE = True
except ImportError:
    JHTDB_AVAILABLE = False
    print("[WARNING] Package givernylocal non détecté. "
          "Les fonctions d'extraction JHTDB ne pourront pas être appelées.")

# --- Dataset / physical constants (fixed for this project) -----------------
DATASET_TITLE = 'isotropic1024coarse'
OUTPUT_PATH = './giverny_output'
TIME_TARGET = 1.0
DT = 0.002
NU = 0.000185


def get_dataset(auth_token=None, output_path=OUTPUT_PATH, dataset_title=DATASET_TITLE):
    """Opens (or reuses) the JHTDB dataset connection.

    auth_token defaults to the JHTDB_AUTH_TOKEN environment variable.
    Get your own token at https://turbulence.pha.jhu.edu/ and put it in a
    local, untracked .env file (see .env.example).
    """
    if not JHTDB_AVAILABLE:
        raise ImportError("givernylocal is not installed.")
    if auth_token is None:
        auth_token = os.environ.get("JHTDB_AUTH_TOKEN")
        if not auth_token:
            raise RuntimeError(
                "No JHTDB auth token found. Set the JHTDB_AUTH_TOKEN "
                "environment variable (see .env.example) or pass "
                "auth_token= explicitly."
            )
    return turb_dataset(dataset_title=dataset_title, output_path=output_path,
                         auth_token=auth_token)


def extract_gradient_snapshot(dataset, N, t=None, dt=DT, nu=NU):
    """Extracts dPdx, dPdz (via the Navier-Stokes momentum equation) and the
    reference DNS pressure, on a centered XZ plane of the JHTDB cube.

    Used from Section 5 onward wherever a clean (noise-free) pressure-
    gradient field is needed as the noise-injection baseline.

    Returns
    -------
    dPdx_clean, dPdz_clean, P_exact_centered, h
    """
    if t is None:
        t = TIME_TARGET
    x_coords = np.linspace(0.0, 0.5, N, dtype=np.float64)
    z_coords = np.linspace(0.0, 0.5, N, dtype=np.float64)
    y_target = np.pi
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_coords, y_target, z_coords, indexing='ij')
    points = np.array([X_mesh.ravel(), Y_mesh.ravel(), Z_mesh.ravel()], dtype=np.float64).T

    result_vel = getData(dataset, 'velocity', t, 'none', 'lag8', 'field', points)
    result_vel_prev = getData(dataset, 'velocity', t - dt, 'none', 'lag8', 'field', points)
    result_vel_next = getData(dataset, 'velocity', t + dt, 'none', 'lag8', 'field', points)
    result_grad = getData(dataset, 'velocity', t, 'none', 'fd8noint', 'gradient', points)
    result_lap = getData(dataset, 'velocity', t, 'none', 'fd8noint', 'laplacian', points)
    result_pression = getData(dataset, 'pressure', t, 'none', 'lag8', 'field', points)

    vel_array = np.array(result_vel[0])
    vel_next_array = np.array(result_vel_next[0])
    vel_prev_array = np.array(result_vel_prev[0])
    grad_array = np.array(result_grad[0])
    lap_array = np.array(result_lap[0])

    u_x = vel_array[:, 0].reshape(N, N)
    u_y = vel_array[:, 1].reshape(N, N)
    u_z = vel_array[:, 2].reshape(N, N)
    u_x_prev = vel_prev_array[:, 0].reshape(N, N)
    u_z_prev = vel_prev_array[:, 2].reshape(N, N)
    u_x_next = vel_next_array[:, 0].reshape(N, N)
    u_z_next = vel_next_array[:, 2].reshape(N, N)

    du_x_dt = (u_x_next - u_x_prev) / (2 * dt)
    du_z_dt = (u_z_next - u_z_prev) / (2 * dt)

    du_x_dx = grad_array[:, 0].reshape(N, N)
    du_x_dy = grad_array[:, 1].reshape(N, N)
    du_x_dz = grad_array[:, 2].reshape(N, N)
    du_z_dx = grad_array[:, 6].reshape(N, N)
    du_z_dy = grad_array[:, 7].reshape(N, N)
    du_z_dz = grad_array[:, 8].reshape(N, N)

    lap_ux = lap_array[:, 0].reshape(N, N)
    lap_uz = lap_array[:, 2].reshape(N, N)

    dPdx_clean = -(du_x_dt + u_x * du_x_dx + u_y * du_x_dy + u_z * du_x_dz) + nu * lap_ux
    dPdz_clean = -(du_z_dt + u_x * du_z_dx + u_y * du_z_dy + u_z * du_z_dz) + nu * lap_uz

    P_exact = np.array(result_pression[0]).reshape(N, N)
    P_exact_centered = P_exact - P_exact.mean()

    h = 0.5 / (N - 1)
    return dPdx_clean, dPdz_clean, P_exact_centered, h


def extract_velocity_snapshot(dataset, N, t=TIME_TARGET, dt=DT):
    """Extracts the raw velocity components u_x, u_z (plus u_y and the
    out-of-plane gradients, and the DNS reference pressure) needed to
    inject noise directly at the velocity level (Section 7), rather than
    at the pressure-gradient level (Section 6).

    Returns a dict with keys:
    u_x, u_y, u_z, u_x_prev, u_z_prev, u_x_next, u_z_next,
    du_x_dy, du_z_dy, P_exact_centered, h
    """
    x_coords = np.linspace(0.0, 0.5, N, dtype=np.float64)
    z_coords = np.linspace(0.0, 0.5, N, dtype=np.float64)
    y_target = np.pi
    X_mesh, Y_mesh, Z_mesh = np.meshgrid(x_coords, y_target, z_coords, indexing='ij')
    points = np.array([X_mesh.ravel(), Y_mesh.ravel(), Z_mesh.ravel()], dtype=np.float64).T

    result_vel = getData(dataset, 'velocity', t, 'none', 'lag8', 'field', points)
    result_vel_prev = getData(dataset, 'velocity', t - dt, 'none', 'lag8', 'field', points)
    result_vel_next = getData(dataset, 'velocity', t + dt, 'none', 'lag8', 'field', points)
    result_grad = getData(dataset, 'velocity', t, 'none', 'fd8noint', 'gradient', points)
    result_pression = getData(dataset, 'pressure', t, 'none', 'lag8', 'field', points)

    vel_array = np.array(result_vel[0])
    vel_next_array = np.array(result_vel_next[0])
    vel_prev_array = np.array(result_vel_prev[0])
    grad_array = np.array(result_grad[0])

    u_x = vel_array[:, 0].reshape(N, N)
    u_y = vel_array[:, 1].reshape(N, N)
    u_z = vel_array[:, 2].reshape(N, N)
    u_x_prev = vel_prev_array[:, 0].reshape(N, N)
    u_z_prev = vel_prev_array[:, 2].reshape(N, N)
    u_x_next = vel_next_array[:, 0].reshape(N, N)
    u_z_next = vel_next_array[:, 2].reshape(N, N)

    du_x_dy = grad_array[:, 1].reshape(N, N)
    du_z_dy = grad_array[:, 7].reshape(N, N)

    P_exact = np.array(result_pression[0]).reshape(N, N)
    P_exact_centered = P_exact - P_exact.mean()

    h = 0.5 / (N - 1)
    return dict(u_x=u_x, u_y=u_y, u_z=u_z, u_x_prev=u_x_prev, u_z_prev=u_z_prev,
                u_x_next=u_x_next, u_z_next=u_z_next, du_x_dy=du_x_dy, du_z_dy=du_z_dy,
                P_exact_centered=P_exact_centered, h=h)
