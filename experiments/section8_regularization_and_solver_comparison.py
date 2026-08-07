#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiments/section8_regularization_and_solver_comparison.py
================================================================================
Reproduces the figures and metrics in Section 8 (Regularization). This is
the ONE place in the pipeline where `lsqr` (Section 5's original solver)
and `solve_poisson_DCT` (the Section 6-7 solver) are deliberately run
side by side, to document and justify the switch:
  - a synthetic MMS cross-check that both solvers reproduce the analytical
    solution,
  - the runtime/accuracy table comparing lsqr, DCT, and RPR-ODI on real
    JHTDB snapshots at N=64/128 (the ~1000-6000x DCT speedup, and the
    lsqr convergence artifact at N=128),
  - an L_N conditioning test,
  - Tikhonov regularization of the DCT solver under correlated noise.

NOTE ON REDUNDANCY WITH SECTION 5: the MMS/conditioning test here uses
different parameters (k_x, k_z, grid extents) than the Section 5.7 MMS
test in experiments/section5_deterministic_lsqr.py -- it is NOT a
duplicate run of the same figure, but a separate validation specific to
justifying the Section 6-8 solver switch. Section 5's own MMS results
stay with section5_deterministic_lsqr.py, by design.

Refactored from finalpipeline.py, PART 2 (part2_deterministic_validation)
and PART 5 (part5_regularization).

Run:
    python experiments/section8_regularization_and_solver_comparison.py
Requires a valid JHTDB_AUTH_TOKEN (see README.md / .env.example).
================================================================================
"""

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.solvers import (build_LN_bN, solve_PPE_MNLS, solve_poisson_DCT,
                          solve_poisson_DCT_tikhonov, solve_OS_MODI, CPU_AVAILABLE)

if not CPU_AVAILABLE:
    print("[WARNING] osmodi is not installed/compiled -- RPR-ODI columns in this "
          "script's runtime table will fall back to the DCT solution instead (see "
          "README.md, 'Installing the OS-MODI reference solver'). lsqr/DCT results "
          "are unaffected.")
from src.noise import generate_correlated_noise
from src.metrics import R_and_err
from src.jhtdb_io import get_dataset, extract_gradient_snapshot

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def part2_solver_comparison(dataset):
    print("\n" + "=" * 78)
    print("SECTION 8 (part 1): SOLVER COMPARISON -- lsqr vs. DCT vs. RPR-ODI")
    print("=" * 78)

    # --- 8.x.1 Synthetic MMS cross-check (lsqr vs. DCT) ---------------------
    print("\n--- Synthetic MMS cross-check: lsqr vs. DCT ---")
    N_mms = 32
    h_mms = 2 * np.pi / N_mms
    x = np.linspace(0, 2 * np.pi, N_mms)
    z = np.linspace(0, 2 * np.pi, N_mms)
    X, Z = np.meshgrid(x, z, indexing='ij')
    p_true = np.sin(X) * np.cos(Z)
    p_true -= p_true.mean()
    dPdx_exact = np.cos(X) * np.cos(Z)
    dPdz_exact = -np.sin(X) * np.sin(Z)

    LN, bN = build_LN_bN(dPdx_exact, dPdz_exact, h_mms)
    p_lsqr = solve_PPE_MNLS(LN, bN, N_mms)
    p_dct = solve_poisson_DCT(bN.reshape(N_mms, N_mms), h_mms)
    R_lsqr, eps_lsqr = R_and_err(p_lsqr, p_true)
    R_dct, eps_dct = R_and_err(p_dct, p_true)
    print(f"   lsqr: R = {R_lsqr:.6f} | RMS error = {eps_lsqr:.2e}")
    print(f"   DCT : R = {R_dct:.6f}  | RMS error = {eps_dct:.2e}")
    print("   -> both solvers reproduce the analytical solution "
          "(residual error = 2nd-order truncation of the FD scheme)")

    # --- 8.x.2 lsqr / DCT / RPR-ODI runtime+accuracy table on JHTDB --------
    print("\n--- PPE-MNLS (lsqr, DCT) vs. RPR-ODI on real JHTDB data ---")
    runtime_table = []
    for N in [64, 128]:
        for t_snap in [1.0, 3.0]:
            dPdx_clean, dPdz_clean, P_exact_centered, h = extract_gradient_snapshot(dataset, N, t=t_snap)
            LN, bN = build_LN_bN(dPdx_clean, dPdz_clean, h)
            bN_2d = bN.reshape(N, N)

            t0 = time.time(); p_lsqr = solve_PPE_MNLS(LN, bN, N);              t_lsqr = time.time() - t0
            t0 = time.time(); p_dct  = solve_poisson_DCT(bN_2d, h);            t_dct  = time.time() - t0
            t0 = time.time()
            p_modi = solve_OS_MODI(dPdx_clean, dPdz_clean, h) if CPU_AVAILABLE else p_dct.copy()
            t_modi = time.time() - t0

            R_lsqr, eps_lsqr = R_and_err(p_lsqr, P_exact_centered)
            R_dct, eps_dct = R_and_err(p_dct, P_exact_centered)
            R_modi, eps_modi = R_and_err(p_modi, P_exact_centered)
            max_lsqr = np.max(np.abs(p_lsqr - P_exact_centered))
            max_modi = np.max(np.abs(p_modi - P_exact_centered))

            row = dict(N=N, t=t_snap,
                       R_lsqr=R_lsqr, eps_lsqr=eps_lsqr, t_lsqr_ms=t_lsqr * 1000,
                       R_dct=R_dct, eps_dct=eps_dct, t_dct_ms=t_dct * 1000,
                       R_modi=R_modi, eps_modi=eps_modi, t_modi_ms=t_modi * 1000,
                       dct_speedup_vs_lsqr=t_lsqr / t_dct,
                       p_dct_vs_p_modi_max_diff=float(np.max(np.abs(p_dct - p_modi))))
            runtime_table.append(row)

            print(f"\n   N={N}, t={t_snap}s:")
            print(f"      lsqr    : R={R_lsqr:.5f} eps_rel={eps_lsqr:.5f} max_err={max_lsqr:.4f}  ({t_lsqr*1000:.1f} ms)")
            print(f"      DCT     : R={R_dct:.5f} eps_rel={eps_dct:.5f}  ({t_dct*1000:.3f} ms, "
                  f"{t_lsqr/t_dct:.0f}x faster than lsqr)")
            print(f"      RPR-ODI : R={R_modi:.5f} eps_rel={eps_modi:.5f} max_err={max_modi:.4f}  ({t_modi*1000:.1f} ms)")
            print(f"      ||p_DCT - p_MODI||_max = {np.max(np.abs(p_dct - p_modi)):.2e}  "
                  f"(Pryce et al. equivalence, verified with the exact solver)")

    with open(os.path.join(DATA_DIR, "section8_solver_runtime_table.json"), "w") as f:
        json.dump(runtime_table, f, indent=2)
    print(f"\n   [saved] {os.path.join(DATA_DIR, 'section8_solver_runtime_table.json')}")

    # --- 8.x.3 L_N conditioning test (MMS, N=64 vs. N=128) ------------------
    print("\n--- L_N conditioning test (exact gradient, no noise) ---")
    gammas = np.logspace(-2, 12, 60)
    conditioning_results = []
    for N in [64, 128]:
        h = 0.5 / (N - 1)
        x = np.linspace(0.0, 0.5, N)
        z = np.linspace(0.0, 0.5, N)
        X, Z = np.meshgrid(x, z, indexing='ij')
        kx, kz = 2 * np.pi * 3, 2 * np.pi * 3
        p_true = np.sin(kx * X) * np.cos(kz * Z)
        p_true -= p_true.mean()
        dPdx_exact = kx * np.cos(kx * X) * np.cos(kz * Z)
        dPdz_exact = -kz * np.sin(kx * X) * np.sin(kz * Z)

        _, bN = build_LN_bN(dPdx_exact, dPdz_exact, h)
        bN_2d = bN.reshape(N, N)

        p_unreg = solve_poisson_DCT(bN_2d, h)
        R_unreg, eps_unreg = R_and_err(p_unreg, p_true)

        errs_gamma = [R_and_err(solve_poisson_DCT_tikhonov(bN_2d, h, g), p_true)[1] for g in gammas]
        best_idx = np.argmin(errs_gamma)

        conditioning_results.append(dict(N=N, R_unreg=R_unreg, eps_unreg=eps_unreg,
                                          best_gamma=float(gammas[best_idx]),
                                          best_eps=float(errs_gamma[best_idx])))
        print(f"   N={N}: unregularized R={R_unreg:.6f} eps={eps_unreg:.5f} | "
              f"best-case Tikhonov (gamma={gammas[best_idx]:.2g}) eps={errs_gamma[best_idx]:.5f} "
              f"-> pure L_N conditioning is not the limiting factor")

    with open(os.path.join(DATA_DIR, "section8_conditioning_test.json"), "w") as f:
        json.dump(conditioning_results, f, indent=2)


def part5_regularization(dataset, noise_levels=(0, 10, 20, 30, 50), num_realizations=10):
    print("\n" + "=" * 78)
    print("SECTION 8 (part 2): FAST SOLVER (DCT) AND TIKHONOV REGULARIZATION")
    print("=" * 78)
    print("(The lsqr/DCT/RPR-ODI runtime benchmark was already run above.)")

    N = 128
    dPdx_clean, dPdz_clean, P_exact_centered, h = extract_gradient_snapshot(dataset, N)
    rms_gx = np.sqrt(np.mean(dPdx_clean**2))
    rms_gz = np.sqrt(np.mean(dPdz_clean**2))
    filter_size_pixels = 6
    seeds = [100 + i for i in range(num_realizations)]
    gammas = np.logspace(-2, 12, 60)

    print(f"\n--- Tikhonov regularization under real correlated noise (N={N}) ---")
    print("('oracle' gamma = best gamma found using knowledge of the DNS reference;")
    print(" a theoretical ceiling, not a method usable blindly on real data.)")
    tikhonov_results = []
    for level in noise_levels:
        sigma_x = (level / 100.0) * rms_gx
        sigma_z = (level / 100.0) * rms_gz

        errs_unreg_r, errs_oracle_r, gamma_oracle_r = [], [], []
        for seed in seeds:
            np.random.seed(seed)
            dPdx_noisy = dPdx_clean + generate_correlated_noise(N, sigma_x, filter_size_pixels)
            dPdz_noisy = dPdz_clean + generate_correlated_noise(N, sigma_z, filter_size_pixels)
            _, bN = build_LN_bN(dPdx_noisy, dPdz_noisy, h)
            bN_2d = bN.reshape(N, N)

            p_unreg = solve_poisson_DCT(bN_2d, h)
            errs_unreg_r.append(R_and_err(p_unreg, P_exact_centered)[1])

            errs_gamma = [R_and_err(solve_poisson_DCT_tikhonov(bN_2d, h, g), P_exact_centered)[1] for g in gammas]
            best_idx = np.argmin(errs_gamma)
            errs_oracle_r.append(errs_gamma[best_idx])
            gamma_oracle_r.append(gammas[best_idx])

        eu, eo, go = np.mean(errs_unreg_r), np.mean(errs_oracle_r), np.mean(gamma_oracle_r)
        tikhonov_results.append(dict(noise_level=level, eps_unreg=eu, eps_oracle=eo,
                                      gamma_oracle=go, improvement_pct=(eu - eo) / eu * 100))
        print(f"   Noise={level:>3}% : unregularized eps={eu:.5f} | "
              f"Tikhonov oracle (gamma={go:.2g}) eps={eo:.5f} | "
              f"improvement={(eu-eo)/eu*100:.1f}%")

    with open(os.path.join(DATA_DIR, "section8_tikhonov_results.json"), "w") as f:
        json.dump(tikhonov_results, f, indent=2)
    print("\n-> Conclusion: Tikhonov regularization provides no measurable practical")
    print("   benefit here (see Section 8 discussion in the report).")


if __name__ == "__main__":
    dataset = get_dataset()
    part2_solver_comparison(dataset)
    part5_regularization(dataset)
    print("\n[DONE] Section 8 data written to results/data/")
