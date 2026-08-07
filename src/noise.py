#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/noise.py
================================================================================
Spatially correlated synthetic noise generators.

IMPORTANT: generate_correlated_noise and generate_correlated_noise_centered
are NOT interchangeable -- see each docstring. Using the wrong one for the
wrong section was an earlier bug during development (Section 7.1
methodological note); this module keeps them clearly separated and named
after the section that uses each one to prevent a regression.
================================================================================
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def generate_correlated_noise(N, scale, filter_size_pixels):
    """Spatially correlated Gaussian noise field, used for PRESSURE-GRADIENT
    noise injection (Section 6). Does NOT remove the sample mean of the
    realization.
    """
    raw = np.random.normal(0, 1, size=(N, N))
    filtered = gaussian_filter(raw, sigma=filter_size_pixels)
    std = np.std(filtered)
    return (filtered / std) * scale if std > 0 else filtered


def generate_correlated_noise_centered(N, scale, filter_size_pixels):
    """Spatially correlated Gaussian noise field, used for VELOCITY noise
    injection (Section 7). A single realization of a spatially filtered
    Gaussian field does not have exactly zero spatial mean over a finite
    N x N grid; this residual offset was found to bias noise-injection
    diagnostics, growing with correlation length. Removing the sample mean
    of each realization before normalization corrects this.

    Do not substitute generate_correlated_noise here: the two functions are
    calibrated to different report sections and are not equivalent.
    """
    raw = np.random.normal(0, 1, size=(N, N))
    filtered = gaussian_filter(raw, sigma=filter_size_pixels)
    filtered = filtered - filtered.mean()  # remove the realization's DC offset
    std = np.std(filtered)
    return (filtered / std) * scale if std > 0 else filtered
