#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/section7_velocity_noise_dct.py
================================================================================
Reproduces the figures and metrics in Section 7 (Velocity-Level Noise
Propagation / Differentiation Amplification), including the core-insight
diagnostic: the unsteady acceleration term du/dt = (u^{n+1}-u^{n-1})/(2dt)
amplifies velocity noise by ~1/(sqrt(2)*dt) ~ 353.55x, so that a modest
velocity noise level produces a much larger noise-to-signal ratio once it
reaches the pressure solvers.

Refactored from finalpipeline.py, PART 4 (part4_velocity_noise).

Run:
    python experiments/section7_velocity_noise_dct.py
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
from src.noise import generate_correlated_noise_centered
from src.metrics import R_and_err, gradients_from_velocity
from src.jhtdb_io import get_dataset, extract_velocity_snapshot, DT, NU

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


def part_velocity_noise(dataset, noise_levels=(0, 2, 5, 10, 20, 30, 50), num_realizations=10):
    print("\n" + "=" * 78)
    print("SECTION 7: VELOCITY-LEVEL NOISE PROPAGATION (TEMPORAL AMPLIFICATION)")
    print("=" * 78)

    # --- 7.1 Amplification diagnostic (theoretical factor 1/(sqrt(2) dt)) --
    print("\n--- 7.1 Theoretical amplification factor of temporal differentiation ---")
    A_theoretical = 1.0 / (np.sqrt(2) * DT)
    print(f"   A = 1/(sqrt(2)*dt) = {A_theoretical:.2f}x  (dt = {DT} s, JHTDB time step)")

    print("\n--- 7.1bis Diagnostic cross-check (N=64, 2% velocity noise) ---")
    fields64 = extract_velocity_snapshot(dataset, 64)
    rms_ux64 = np.sqrt(np.mean(fields64['u_x']**2))
    sigma_test = 0.02 * rms_ux64
    du_x_dt_clean = (fields64['u_x_next'] - fields64['u_x_prev']) / (2 * DT)
    clean_rms = np.sqrt(np.mean(du_x_dt_clean**2))

    ratios = []
    for seed in range(100, 120):  # 20 realizations
        np.random.seed(seed)
        noise_prev = generate_correlated_noise_centered(64, sigma_test, 1.5)
        noise_next = generate_correlated_noise_centered(64, sigma_test, 1.5)
        noise_du_x_dt = (noise_next - noise_prev) / (2 * DT)
        ratios.append(np.sqrt(np.mean(noise_du_x_dt**2)) / clean_rms * 100)
    ratios = np.array(ratios)
    print(f"   Clean unsteady accel. RMS       : {clean_rms:.5f} m/s^2")
    print(f"   Velocity noise RMS (2%)         : {sigma_test:.5f} m/s")
    print(f"   Noise-to-signal ratio in du/dt  : {ratios.mean():.2f} +/- {ratios.std():.2f} %")
    print(f"   Theoretical prediction          : {sigma_test/(np.sqrt(2)*DT)/clean_rms*100:.2f} %")

    results = {}
    for N in [64, 128]:
        filter_size_pixels = 1.5 if N == 64 else 10
        fields = extract_velocity_snapshot(dataset, N)
        h = fields['h']
        rms_ux = np.sqrt(np.mean(fields['u_x']**2))
        rms_uz = np.sqrt(np.mean(fields['u_z']**2))
        rms_P_exact = np.sqrt(np.mean(fields['P_exact_centered']**2))
        seeds = [100 + i for i in range(num_realizations)]

        mean_R_ppe, std_R_ppe, mean_eps_ppe, std_eps_ppe = [], [], [], []
        mean_R_modi, std_R_modi, mean_eps_modi, std_eps_modi = [], [], [], []
        mean_diff, std_diff = [], []

        print(f"\n   Grid N={N}")
        for level in noise_levels:
            sigma_ux = (level / 100.0) * rms_ux
            sigma_uz = (level / 100.0) * rms_uz

            R_ppe_r, eps_ppe_r, R_modi_r, eps_modi_r, diff_r = [], [], [], [], []
            for seed in seeds:
                np.random.seed(seed)
                u_x_noisy = fields['u_x'] + generate_correlated_noise_centered(N, sigma_ux, filter_size_pixels)
                u_z_noisy = fields['u_z'] + generate_correlated_noise_centered(N, sigma_uz, filter_size_pixels)
                u_x_prev_noisy = fields['u_x_prev'] + generate_correlated_noise_centered(N, sigma_ux, filter_size_pixels)
                u_z_prev_noisy = fields['u_z_prev'] + generate_correlated_noise_centered(N, sigma_uz, filter_size_pixels)
                u_x_next_noisy = fields['u_x_next'] + generate_correlated_noise_centered(N, sigma_ux, filter_size_pixels)
                u_z_next_noisy = fields['u_z_next'] + generate_correlated_noise_centered(N, sigma_uz, filter_size_pixels)

                dPdx_noisy, dPdz_noisy = gradients_from_velocity(
                    u_x_noisy, fields['u_y'], u_z_noisy, u_x_prev_noisy, u_z_prev_noisy,
                    u_x_next_noisy, u_z_next_noisy, h, DT, NU, fields['du_x_dy'], fields['du_z_dy'])

                _, bN = build_LN_bN(dPdx_noisy, dPdz_noisy, h)
                p_ppe = solve_poisson_DCT(bN.reshape(N, N), h)
                p_modi = solve_OS_MODI(dPdx_noisy, dPdz_noisy, h) if CPU_AVAILABLE else p_ppe.copy()

                R, eps = R_and_err(p_ppe, fields['P_exact_centered']);  R_ppe_r.append(R);  eps_ppe_r.append(eps)
                R, eps = R_and_err(p_modi, fields['P_exact_centered']); R_modi_r.append(R); eps_modi_r.append(eps)
                diff_r.append(np.sqrt(np.mean((p_ppe - p_modi)**2)) / rms_P_exact)

            mean_R_ppe.append(np.mean(R_ppe_r));   std_R_ppe.append(np.std(R_ppe_r))
            mean_eps_ppe.append(np.mean(eps_ppe_r)); std_eps_ppe.append(np.std(eps_ppe_r))
            mean_R_modi.append(np.mean(R_modi_r)); std_R_modi.append(np.std(R_modi_r))
            mean_eps_modi.append(np.mean(eps_modi_r)); std_eps_modi.append(np.std(eps_modi_r))
            mean_diff.append(np.mean(diff_r));     std_diff.append(np.std(diff_r))
            print(f"      Velocity noise={level:>3}% : PPE-MNLS R={mean_R_ppe[-1]:.4f} eps={mean_eps_ppe[-1]:.3f} | "
                  f"RPR-ODI R={mean_R_modi[-1]:.4f} eps={mean_eps_modi[-1]:.3f} | "
                  f"||p_PPE-p_ODI||_rel={mean_diff[-1]:.5f}")

        results[N] = dict(noise_levels=list(noise_levels),
                           R_ppe=mean_R_ppe, R_ppe_std=std_R_ppe,
                           eps_ppe=mean_eps_ppe, eps_ppe_std=std_eps_ppe,
                           R_modi=mean_R_modi, R_modi_std=std_R_modi,
                           eps_modi=mean_eps_modi, eps_modi_std=std_eps_modi,
                           diff=mean_diff, diff_std=std_diff)

        # --- Triptych plot ---
        r = results[N]
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.errorbar(r['noise_levels'], r['R_ppe'], yerr=r['R_ppe_std'], fmt='o-', color='crimson',
                     linewidth=2, elinewidth=1.5, capsize=4, label='PPE-MNLS')
        plt.errorbar(r['noise_levels'], r['R_modi'], yerr=r['R_modi_std'], fmt='s--', color='royalblue',
                     linewidth=2, elinewidth=1.5, capsize=4, label='RPR-ODI')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Spatial Correlation vs.\\ Velocity Noise\n($N={N}$)', fontsize=10, fontweight='bold')
        plt.xlabel('Velocity Noise Level (%)', fontsize=10)
        plt.ylabel('Correlation with DNS Pressure', fontsize=10)
        plt.legend(fontsize=9)

        plt.subplot(1, 3, 2)
        plt.errorbar(r['noise_levels'], r['eps_ppe'], yerr=r['eps_ppe_std'], fmt='o-', color='crimson',
                     linewidth=2, elinewidth=1.5, capsize=4, label='PPE-MNLS')
        plt.errorbar(r['noise_levels'], r['eps_modi'], yerr=r['eps_modi_std'], fmt='s--', color='royalblue',
                     linewidth=2, elinewidth=1.5, capsize=4, label='RPR-ODI')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Relative RMS Error vs.\\ Velocity Noise\n($N={N}$)', fontsize=10, fontweight='bold')
        plt.xlabel('Velocity Noise Level (%)', fontsize=10)
        plt.ylabel('Normalized RMS Error', fontsize=10)
        plt.legend(fontsize=9)

        plt.subplot(1, 3, 3)
        plt.errorbar(r['noise_levels'], r['diff'], yerr=r['diff_std'], fmt='d-', color='purple',
                     linewidth=2, elinewidth=1.5, capsize=4,
                     label=r'$\|p_{PPE} - p_{ODI}\|_{RMS} / \|p_{DNS}\|_{RMS}$')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f'Direct Solver Difference\n($N={N}$)', fontsize=10, fontweight='bold')
        plt.xlabel('Velocity Noise Level (%)', fontsize=10)
        plt.ylabel('Normalized RMS Difference', fontsize=10)
        plt.legend(fontsize=9)

        savefig(f"section7_velocity_noise_N{N}.png")

    with open(os.path.join(DATA_DIR, "section7_velocity_noise_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   [saved] {os.path.join(DATA_DIR, 'section7_velocity_noise_results.json')}")

    return results


if __name__ == "__main__":
    dataset = get_dataset()
    part_velocity_noise(dataset)
    print("\n[DONE] Section 7 figures/data written to results/")
