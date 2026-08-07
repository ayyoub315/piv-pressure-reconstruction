#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py
================================================================================
Runs every experiment script in the same order as the report sections, and
regenerates every figure/data file under results/. Requires JHTDB_AUTH_TOKEN
to be set (see README.md / .env.example).

Run a single section instead of everything with, e.g.:
    python experiments/section6_gradient_noise_dct.py
================================================================================
"""

import subprocess
import sys
import os

SCRIPTS = [
    "experiments/section5_deterministic_lsqr.py",
    "experiments/section6_gradient_noise_dct.py",
    "experiments/section7_velocity_noise_dct.py",
    "experiments/section8_regularization_and_solver_comparison.py",
]

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    for script in SCRIPTS:
        print("\n" + "#" * 78)
        print(f"# RUNNING {script}")
        print("#" * 78)
        result = subprocess.run([sys.executable, os.path.join(root, script)])
        if result.returncode != 0:
            print(f"\n[ERROR] {script} exited with code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

    print("\n[DONE] All sections reproduced. See results/figures/ and results/data/.")
