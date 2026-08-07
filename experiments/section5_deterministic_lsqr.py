#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/section5_deterministic_lsqr.py
================================================================================
Reproduces every figure and printed metric in Section 5 (Deterministic
Results) of the report. This is the ORIGINAL solver configuration used for
this project: PPE-MNLS solved via sparse `lsqr`, before the DCT solver was
adopted from Section 6 onward (see src/solvers.py docstring and Section 8
of the report for that transition and its justification).

Refactored from the legacy Spyder script `section5rapportfinal.py`:
  - solver/noise/metric code now imported from src/ (no duplication)
  - plt.show() replaced with plt.savefig(...) so every figure is written to
    results/figures/ under the exact filename referenced by
    \\includegraphics in 05_deterministic_results.tex

Run:
    python experiments/section5_deterministic_lsqr.py
Requires a valid JHTDB_AUTH_TOKEN (see README.md / .env.example).
================================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.solvers import build_LN_bN, solve_PPE_MNLS, solve_OS_MODI, CPU_AVAILABLE
from src.jhtdb_io import get_dataset, extract_gradient_snapshot, JHTDB_AVAILABLE

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [saved] {path}")


# =============================================================================
# 5.1 Synthetic test case (32x32)
# =============================================================================

def run_section_5_1_synthetic():
    print("\n--- [5.1] Synthetic Test Case (32x32) ---")
    N = 32
    L = 2 * np.pi
    h = L / N

    x = np.linspace(0, L - h, N)
    y = np.linspace(0, L - h, N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    P_exact = np.sin(X) * np.cos(Y)
    P_exact_centered = P_exact - P_exact.mean()

    dPdx = np.cos(X) * np.cos(Y)
    dPdy = -np.sin(X) * np.sin(Y)

    LN, bN = build_LN_bN(dPdx, dPdy, h)
    P_ppe = solve_PPE_MNLS(LN, bN, N)
    error = np.abs(P_ppe - P_exact_centered)

    print(f"Max Error  : {error.max():.6e}")
    print(f"Mean Error : {error.mean():.6e}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, P_exact_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Exact P')

    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, P_ppe, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title('Reconstructed P (PPE-MNLS)')

    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, error, levels=50, cmap='hot')
    plt.colorbar()
    plt.title('Absolute Error')

    savefig("testsolveurPPE.png")


# =============================================================================
# 5.2 - 5.6 & 5.8 : full JHTDB pipeline at a given N
# =============================================================================

def run_section_5_jhtdb_pipeline(dataset, N=64, t_target=1.0):
    if not JHTDB_AVAILABLE:
        print("[SKIP] givernylocal unavailable.")
        return

    print(f"\n--- [JHTDB] Processing N={N}, t={t_target}s ---")
    dPdx, dPdz, P_exact_centered, h = extract_gradient_snapshot(dataset, N, t=t_target)

    x_coords = np.linspace(0.0, 0.5, N, dtype=np.float64)
    z_coords = np.linspace(0.0, 0.5, N, dtype=np.float64)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')

    # --- 1. Gradient verification plot (Section 5.6) ---
    # Re-derive the DNS-consistent gradient from P_exact for comparison.
    dPdx_dns, dPdz_dns = np.gradient(P_exact_centered, h)
    error_grad_x = np.abs(dPdx - dPdx_dns)
    error_grad_z = np.abs(dPdz - dPdz_dns)
    error_grad_magnitude = np.sqrt(error_grad_x**2 + error_grad_z**2)

    print(f"\n================ GRADIENT VERIFICATION (N={N}) ================")
    print(f"RMS Error on dP/dx   : {np.sqrt(np.mean(error_grad_x**2)):.6e}")
    print(f"RMS Error on dP/dz   : {np.sqrt(np.mean(error_grad_z**2)):.6e}")
    print(f"Mean Error Magnitude : {np.mean(error_grad_magnitude):.6e}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Z, error_grad_x, levels=50, cmap='inferno')
    plt.colorbar()
    plt.title(r'Local Error on X-Gradient: $|\frac{\partial p}{\partial x}_{\mathrm{NS}} - \frac{\partial p}{\partial x}_{\mathrm{DNS}}|$')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(1, 2, 2)
    plt.contourf(X, Z, error_grad_magnitude, levels=50, cmap='inferno')
    plt.colorbar()
    plt.title('Total Gradient Error Magnitude')
    plt.xlabel('X')
    plt.ylabel('Z')
    savefig(f"graderror{N}.png")

    # --- Solve with both solvers ---
    LN, bN = build_LN_bN(dPdx, dPdz, h)
    P_ppe_centered, runtime_ppe = solve_PPE_MNLS(LN, bN, N, return_runtime=True)

    if CPU_AVAILABLE:
        P_modi_centered, runtime_modi = solve_OS_MODI(dPdx, dPdz, h, return_runtime=True)
    else:
        P_modi_centered, runtime_modi = P_ppe_centered.copy(), 0.0

    # --- 2. PPE-MNLS 3-panel plot ---
    error_ppe = np.abs(P_exact_centered - P_ppe_centered)
    corr_ppe = np.corrcoef(P_exact_centered.flatten(), P_ppe_centered.flatten())[0, 1]

    dPrecon_ppe_dx, dPrecon_ppe_dz = np.gradient(P_ppe_centered, h)
    res_x_ppe = np.sqrt(np.mean((dPrecon_ppe_dx - dPdx)**2))
    res_z_ppe = np.sqrt(np.mean((dPrecon_ppe_dz - dPdz)**2))
    grad_res_ppe = (res_x_ppe + res_z_ppe) / 2

    print(f"\nPPE-MNLS Validation Metrics (t={t_target} | N={N}) :")
    print(f"Max Error          : {np.max(error_ppe):.6f}")
    print(f"Mean Error         : {np.mean(error_ppe):.6f}")
    print(f"Spatial Correlation: {corr_ppe:.6f}")
    print(f"Solver Runtime     : {runtime_ppe * 1000:.2f} ms")
    print(f"Gradient Residual  : {grad_res_ppe:.6e}")

    plt.figure(figsize=(13, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Z, P_exact_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'Exact P (t={t_target})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 3, 2)
    plt.contourf(X, Z, P_ppe_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'Reconstructed P (R={corr_ppe:.4f})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 3, 3)
    plt.contourf(X, Z, error_ppe, levels=50, cmap='hot')
    plt.colorbar()
    plt.title(f'Absolute Error (Mean={np.mean(error_ppe):.4f})')
    plt.xlabel('X'); plt.ylabel('Z')
    savefig(f"t{int(t_target)}_thi_sparse.png")

    # --- 3. OS-MODI 3-panel plot ---
    error_modi = np.abs(P_exact_centered - P_modi_centered)
    corr_modi = np.corrcoef(P_exact_centered.flatten(), P_modi_centered.flatten())[0, 1]

    dPrecon_modi_dx, dPrecon_modi_dz = np.gradient(P_modi_centered, h)
    res_x_modi = np.sqrt(np.mean((dPrecon_modi_dx - dPdx)**2))
    res_z_modi = np.sqrt(np.mean((dPrecon_modi_dz - dPdz)**2))
    grad_res_modi = (res_x_modi + res_z_modi) / 2

    print(f"\nOfficial OS-MODI Validation Metrics (t={t_target} | N={N}) :")
    print(f"Max Error          : {np.max(error_modi):.6f}")
    print(f"Mean Error         : {np.mean(error_modi):.6f}")
    print(f"Spatial Correlation: {corr_modi:.6f}")
    print(f"Solver Runtime     : {runtime_modi * 1000:.2f} ms")
    print(f"Gradient Residual  : {grad_res_modi:.6e}")

    plt.figure(figsize=(13, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Z, P_exact_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'Exact P (t={t_target})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 3, 2)
    plt.contourf(X, Z, P_modi_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'Official OS-MODI P (R={corr_modi:.4f})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 3, 3)
    plt.contourf(X, Z, error_modi, levels=50, cmap='hot')
    plt.colorbar()
    plt.title(f'Absolute Error (Mean={np.mean(error_modi):.4f})')
    plt.xlabel('X'); plt.ylabel('Z')
    savefig(f"OSMODI_t{int(t_target)}_N{N}.png")

    # --- 4. Direct difference field (PPE vs MODI) ---
    diff_field = P_ppe_centered - P_modi_centered

    print(f"\n================ METRICS COMPARISON (N={N}) ================")
    print(f"Max Direct Difference (P_ppe - P_modi): {np.max(np.abs(diff_field)):.6e}")
    print(f"Mean Direct Difference                : {np.mean(np.abs(diff_field)):.6e}")

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Z, P_ppe_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'PPE-MNLS Field (N={N})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 3, 2)
    plt.contourf(X, Z, P_modi_centered, levels=50, cmap='RdBu_r')
    plt.colorbar()
    plt.title(f'OS-MODI Field (N={N})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 3, 3)
    max_diff = np.max(np.abs(diff_field))
    plt.contourf(X, Z, diff_field, levels=50, cmap='bwr', vmin=-max_diff, vmax=max_diff)
    plt.colorbar()
    plt.title(r'Direct Difference ($p_{\mathrm{PPE}} - p_{\mathrm{ODI}}$)')
    plt.xlabel('X'); plt.ylabel('Z')
    savefig(f"PPEvsMODI{N}.png")

    # --- 5. Local gradient-residual field plots (Section 5.8) ---
    residual_field_ppe = np.sqrt((dPrecon_ppe_dx - dPdx)**2 + (dPrecon_ppe_dz - dPdz)**2)
    residual_field_odi = np.sqrt((dPrecon_modi_dx - dPdx)**2 + (dPrecon_modi_dz - dPdz)**2)

    print(f"\n--- JHTDB Gradient Residual Means (N={N}) ---")
    print(f"Mean Gradient Residual [PPE JHTDB]  : {np.mean(residual_field_ppe):.4f}")
    print(f"Mean Gradient Residual [MODI JHTDB] : {np.mean(residual_field_odi):.4f}")

    plt.figure(figsize=(12, 5))
    max_res = max(np.max(residual_field_ppe), np.max(residual_field_odi))

    plt.subplot(1, 2, 1)
    plt.contourf(X, Z, residual_field_ppe, levels=50, cmap='viridis', vmin=0, vmax=max_res)
    plt.colorbar(label=r'$|\nabla p_{\mathrm{PPE}} - \mathbf{g}|$')
    plt.title(f'PPE Gradient Residual Field\nJHTDB (N={N})')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(1, 2, 2)
    plt.contourf(X, Z, residual_field_odi, levels=50, cmap='viridis', vmin=0, vmax=max_res)
    plt.colorbar(label=r'$|\nabla p_{\mathrm{ODI}} - \mathbf{g}|$')
    plt.title(f'OS-MODI Gradient Residual Field\nJHTDB (N={N})')
    plt.xlabel('X'); plt.ylabel('Z')
    savefig(f"gradresidualfieldTHI{N}.png")


# =============================================================================
# 5.7 MMS conditioning test at N=128
# =============================================================================

def run_section_5_7_mms_conditioning(N=128):
    print(f"\n--- [5.7] MMS Conditioning Test (N={N}) ---")
    L = 0.5
    x_coords = np.linspace(0.0, L, N, dtype=np.float64)
    z_coords = np.linspace(0.0, L, N, dtype=np.float64)
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
    h = L / (N - 1)

    k_x = 2 * np.pi / 0.4
    k_z = 2 * np.pi / 0.3

    P_exact = np.sin(k_x * X) * np.cos(k_z * Z)
    P_exact_centered = P_exact - P_exact.mean()

    dPdx_exact = k_x * np.cos(k_x * X) * np.cos(k_z * Z)
    dPdz_exact = -k_z * np.sin(k_x * X) * np.sin(k_z * Z)

    LN, bN = build_LN_bN(dPdx_exact, dPdz_exact, h)
    P_ppe_centered = solve_PPE_MNLS(LN, bN, N)

    if CPU_AVAILABLE:
        P_modi_centered = solve_OS_MODI(dPdx_exact, dPdz_exact, h)
    else:
        P_modi_centered = P_ppe_centered.copy()

    diff_field = P_ppe_centered - P_modi_centered
    dPpe_dx, dPpe_dz = np.gradient(P_ppe_centered, h)
    dModi_dx, dModi_dz = np.gradient(P_modi_centered, h)

    res_grad_ppe = np.sqrt((dPpe_dx - dPdx_exact)**2 + (dPpe_dz - dPdz_exact)**2)
    res_grad_modi = np.sqrt((dModi_dx - dPdx_exact)**2 + (dModi_dz - dPdz_exact)**2)

    print(f"PPE Correlation to Exact     : {np.corrcoef(P_exact_centered.flatten(), P_ppe_centered.flatten())[0,1]:.6f}")
    print(f"MODI Correlation to Exact    : {np.corrcoef(P_exact_centered.flatten(), P_modi_centered.flatten())[0,1]:.6f}")
    print(f"Max Direct Difference (P-P)  : {np.max(np.abs(diff_field)):.6e}")
    print(f"Mean Gradient Residual [PPE]  : {np.mean(res_grad_ppe):.6e}")
    print(f"Mean Gradient Residual [MODI] : {np.mean(res_grad_modi):.6e}")

    plt.figure(figsize=(15, 9))
    plt.subplot(2, 2, 1)
    max_diff = np.max(np.abs(diff_field))
    plt.contourf(X, Z, diff_field, levels=50, cmap='bwr', vmin=-max_diff, vmax=max_diff)
    plt.colorbar()
    plt.title(r'Direct Difference Field ($p_{\mathrm{PPE}} - p_{\mathrm{ODI}}$)')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(2, 2, 2)
    plt.contourf(X, Z, np.abs(P_exact_centered - P_ppe_centered), levels=50, cmap='hot')
    plt.colorbar()
    plt.title('PPE Absolute Error vs Exact')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(2, 2, 3)
    plt.contourf(X, Z, res_grad_ppe, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('PPE Local Gradient Residual Field')
    plt.xlabel('X'); plt.ylabel('Z')

    plt.subplot(2, 2, 4)
    plt.contourf(X, Z, res_grad_modi, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('OS-MODI Local Gradient Residual Field')
    plt.xlabel('X'); plt.ylabel('Z')

    savefig(f"MMS{N}.png")


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # 1. Synthetic case (Section 5.1) -- no JHTDB connection needed
    run_section_5_1_synthetic()

    # 2. Full JHTDB pipeline, N=64 (Sections 5.2-5.6 & 5.8)
    if JHTDB_AVAILABLE:
        dataset = get_dataset()
        run_section_5_jhtdb_pipeline(dataset, N=64, t_target=1.0)
    else:
        print("[SKIP] JHTDB pipeline (givernylocal unavailable).")

    # 3. MMS conditioning test at N=128 (Section 5.7)
    run_section_5_7_mms_conditioning(N=128)

    print("\n[DONE] Section 5 figures written to results/figures/")
