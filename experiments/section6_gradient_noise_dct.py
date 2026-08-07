#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/section6_gradient_noise_dct.py
================================================================================
Reproduces the figures and metrics in Section 6 (Sensitivity to Correlated
Pressure-Gradient Noise). This is the first section to use the fast, exact
DCT solver (solve_poisson_DCT) instead of lsqr -- see src/solvers.py and
Section 8 of the report for the justification of that switch.

Refactored from finalpipeline.py, PART 3 (part3_gradient_noise).

Run:
    python experiments/section6_gradient_noise_dct.py
Requires a valid JHTDB_AUTH_TOKEN (see README.md / .env.example).
================================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.solvers import build_LN_bN, solve_poisson_DCT, solve_OS_MODI, CPU_AVAILABLE

if not CPU_AVAILABLE:
    print("[WARNING] osmodi is not installed/compiled -- RPR-ODI results in this "
          "script will fall back to the PPE-MNLS (DCT) solution instead (see "
          "README.md, 'Installing the OS-MODI reference solver'). PPE-MNLS results "
          "are unaffected.")
from src.noise import generate_correlated_noise
from src.metrics import R_and_err
from src.jhtdb_io import get_dataset, extract_gradient_snapshot

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"   [saved] {path}")


def part_gradient_noise(dataset, noise_levels=(0, 5, 10, 20, 30, 50), num_realizations=10):
    print("\n" + "=" * 78)
    print("SECTION 6: SENSITIVITY TO CORRELATED PRESSURE-GRADIENT NOISE")
    print("=" * 78)

    results = {}
    for N in [64, 128]:
        filter_size_pixels = 3 if N == 64 else 6  # matched physical correlation length
        dPdx_clean, dPdz_clean, P_exact_centered, h = extract_gradient_snapshot(dataset, N)
        rms_gx = np.sqrt(np.mean(dPdx_clean**2))
        rms_gz = np.sqrt(np.mean(dPdz_clean**2))
        rms_P_exact = np.sqrt(np.mean(P_exact_centered**2))
        seeds = [100 + i for i in range(num_realizations)]

        mean_R_ppe, std_R_ppe, mean_eps_ppe, std_eps_ppe = [], [], [], []
        mean_R_modi, std_R_modi, mean_eps_modi, std_eps_modi = [], [], [], []
        mean_diff, std_diff = [], []

        print(f"\n   Grid N={N}, filter={filter_size_pixels}px")
        for level in noise_levels:
            sigma_x = (level / 100.0) * rms_gx
            sigma_z = (level / 100.0) * rms_gz

            R_ppe_r, eps_ppe_r, R_modi_r, eps_modi_r, diff_r = [], [], [], [], []
            for seed in seeds:
                np.random.seed(seed)
                dPdx_noisy = dPdx_clean + generate_correlated_noise(N, sigma_x, filter_size_pixels)
                dPdz_noisy = dPdz_clean + generate_correlated_noise(N, sigma_z, filter_size_pixels)

                _, bN = build_LN_bN(dPdx_noisy, dPdz_noisy, h)
                p_ppe = solve_poisson_DCT(bN.reshape(N, N), h)
                p_modi = solve_OS_MODI(dPdx_noisy, dPdz_noisy, h) if CPU_AVAILABLE else p_ppe.copy()

                R, eps = R_and_err(p_ppe, P_exact_centered);   R_ppe_r.append(R);  eps_ppe_r.append(eps)
                R, eps = R_and_err(p_modi, P_exact_centered);  R_modi_r.append(R); eps_modi_r.append(eps)
                diff_r.append(np.sqrt(np.mean((p_ppe - p_modi)**2)) / rms_P_exact)

            mean_R_ppe.append(np.mean(R_ppe_r));   std_R_ppe.append(np.std(R_ppe_r))
            mean_eps_ppe.append(np.mean(eps_ppe_r)); std_eps_ppe.append(np.std(eps_ppe_r))
            mean_R_modi.append(np.mean(R_modi_r)); std_R_modi.append(np.std(R_modi_r))
            mean_eps_modi.append(np.mean(eps_modi_r)); std_eps_modi.append(np.std(eps_modi_r))
            mean_diff.append(np.mean(diff_r));     std_diff.append(np.std(diff_r))
            print(f"      Noise={level:>3}% : PPE-MNLS R={mean_R_ppe[-1]:.5f} eps={mean_eps_ppe[-1]:.5f} | "
                  f"RPR-ODI R={mean_R_modi[-1]:.5f} eps={mean_eps_modi[-1]:.5f} | "
                  f"||p_PPE-p_ODI||_rel={mean_diff[-1]:.5f}")

        results[N] = dict(noise_levels=list(noise_levels),
                           R_ppe=mean_R_ppe, R_ppe_std=std_R_ppe,
                           eps_ppe=mean_eps_ppe, eps_ppe_std=std_eps_ppe,
                           R_modi=mean_R_modi, R_modi_std=std_R_modi,
                           eps_modi=mean_eps_modi, eps_modi_std=std_eps_modi,
                           diff=mean_diff, diff_std=std_diff)

        # --- Triptych plot (correlation / relative RMS error / direct
        # solver difference) ---
        r = results[N]
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.errorbar(r['noise_levels'], r['R_ppe'], yerr=r['R_ppe_std'], fmt='o-', color='crimson',
                     linewidth=2, elinewidth=1.5, capsize=4, label='PPE-MNLS')
        plt.errorbar(r['noise_levels'], r['R_modi'], yerr=r['R_modi_std'], fmt='s--', color='royalblue',
                     linewidth=2, elinewidth=1.5, capsize=4, label='RPR-ODI')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Spatial Correlation vs.\\ Noise Level\n($N={N}$, $\\sigma={filter_size_pixels}$px)',
                   fontsize=10, fontweight='bold')
        plt.xlabel('Correlated Gradient Noise Level (%)', fontsize=10)
        plt.ylabel('Correlation with DNS Pressure', fontsize=10)
        plt.legend(fontsize=9)

        plt.subplot(1, 3, 2)
        plt.errorbar(r['noise_levels'], r['eps_ppe'], yerr=r['eps_ppe_std'], fmt='o-', color='crimson',
                     linewidth=2, elinewidth=1.5, capsize=4, label='PPE-MNLS')
        plt.errorbar(r['noise_levels'], r['eps_modi'], yerr=r['eps_modi_std'], fmt='s--', color='royalblue',
                     linewidth=2, elinewidth=1.5, capsize=4, label='RPR-ODI')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Relative RMS Error vs.\\ Noise Level\n($N={N}$, $\\sigma={filter_size_pixels}$px)',
                   fontsize=10, fontweight='bold')
        plt.xlabel('Correlated Gradient Noise Level (%)', fontsize=10)
        plt.ylabel('Normalized RMS Error', fontsize=10)
        plt.legend(fontsize=9)

        plt.subplot(1, 3, 3)
        plt.errorbar(r['noise_levels'], r['diff'], yerr=r['diff_std'], fmt='d-', color='purple',
                     linewidth=2, elinewidth=1.5, capsize=4,
                     label=r'$\|p_{PPE} - p_{ODI}\|_{RMS} / \|p_{DNS}\|_{RMS}$')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Direct Solver Difference\n($N={N}$)', fontsize=10, fontweight='bold')
        plt.xlabel('Correlated Gradient Noise Level (%)', fontsize=10)
        plt.ylabel('Normalized RMS Difference', fontsize=10)
        plt.legend(fontsize=9)

        savefig(f"section6_gradient_noise_N{N}.png")

    with open(os.path.join(DATA_DIR, "section6_gradient_noise_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   [saved] {os.path.join(DATA_DIR, 'section6_gradient_noise_results.json')}")

    return results


if __name__ == "__main__":
    dataset = get_dataset()
    part_gradient_noise(dataset)
    print("\n[DONE] Section 6 figures/data written to results/")
