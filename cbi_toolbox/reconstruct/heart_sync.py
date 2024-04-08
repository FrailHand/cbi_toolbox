"""
This module implements synchronization methods to generate a 4D volume from 
data consisting of videos collected at different slices of the data. The
underlying data must be periodic.

The slicing axis is expected to be the second, in the following convention:
[T, Z, X, Y], where the first axis is time.

[1] Liebling, M., et al. *"Four-dimensional cardiac imaging in living embryos via
postacquisition synchronization of nongated slice sequences."* Journal of
biomedical optics 10.5 (2005): 054001-054001
"""

# Copyright (c) 2024 UMONS, https://web.umons.ac.be/
# Written by François Marelli <francois.marelli@umons.ac.be>
#
# This file is part of CBI Toolbox.
#
# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the 3-Clause BSD License.
#
# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# 3-Clause BSD License for more details.
#
# You should have received a copy of the 3-Clause BSD License along
# with CBI Toolbox. If not, see https://opensource.org/licenses/BSD-3-Clause.
#
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
from cbi_toolbox import parallel as cbip

import scipy.fft as fft
import scipy.sparse as sparse
import scipy.linalg as linalg


def heart_shifts_4D(data, weights_k=(1, 1, 1), workers=None):
    """
    Compute the absolute shifts of periodic input slices with respect to the
    first slice using the method from [1].
    It also implements an initial shift correction to reduce the issues caused
    by the nonlinear periodicity constraints in the least squares solver.

    Parameters
    ----------
    data : np.ndarray [T, Z, X, Y]
        The data to be synchronized.
    weights_k : np.ndarray
        The wheights describing the importance of neighbouring slices when
        solving the synchronization problem (see [1]).
        `weights_k[0]` refers to the importance of direct neighbours,
        `weights_k[1]` refers to the neighbours 2 slices away, etc.
        By default (1, 1, 1), equivalent weights for slices up to a distance 3.
    workers : int, optional
        The number of cores to use for parallel processing, by default None (auto)

    Returns
    -------
    np.ndarray (int)
        The computed shifts for each depth slice of the input data.

    """

    weights_k = np.atleast_1d(weights_k)
    max_dk = weights_k.shape[0]

    n_period = data.shape[0]
    n_depth = data.shape[1]

    Q_k_dk = np.empty((n_period, n_depth, max_dk + 1))

    if workers is None:
        workers = cbip.max_workers()

    wfft = fft.rfft(data, axis=0, overwrite_x=False, workers=workers)

    # Compute correlation using FFT
    for delta_k in range(max_dk + 1):
        corr = fft.irfft(
            wfft * np.conj(np.roll(wfft, -delta_k, axis=1)),
            axis=0,
            workers=workers,
            overwrite_x=True,
        )
        Q_k_dk[..., delta_k] = corr.sum((-1, -2))

    del wfft

    S_k_dk = np.argmax(Q_k_dk, axis=0)

    A = []
    s = []
    w = []

    # Correction factor taking into account the initial solution above
    s_corr = []

    A0 = np.zeros((1, n_depth))
    A0[:, 0] = 1

    A.append(A0)
    s.append(np.atleast_1d(0))
    w.append(1)
    s_corr.append(np.atleast_1d(0))

    for delta_k in range(max_dk):
        delta_k += 1
        A_d = sparse.diags_array(
            (1, -1), offsets=(0, delta_k), shape=(n_depth - delta_k, n_depth)
        ).toarray()
        A.append(A_d)
        s.append(S_k_dk[: n_depth - delta_k, delta_k])
        w.extend((weights_k[delta_k - 1],) * (n_depth - delta_k))

        if delta_k != 1:
            s_corr.append(sol_init[:-delta_k] - sol_init[delta_k:])
            continue

        # Only for first neighbour, compute initial solution
        A_ = np.vstack(A)
        s_ = np.hstack(s)

        # Compute an initial solution based on 1 neighbour only
        sol_init, _, _, _ = linalg.lstsq(A_, s_)
        sol_init = np.round((sol_init)).astype(int) % n_period

        s_corr.append(sol_init[:-delta_k] - sol_init[delta_k:])

    if max_dk == 1:
        return sol_init

    A = np.vstack(A)
    s = np.hstack(s)
    W = np.sqrt(np.diag(w))

    s_corr = np.hstack(s_corr)

    # Update the shift matrix with the initial solution found
    # This should minimize the residual shifts to correct, to avoid problems
    # with the periodicity of the problem interfering with the least squares
    s = (s - s_corr + n_period // 2) % n_period - n_period // 2

    # Apply the weights
    A = W @ A
    s = s @ W

    sol, _, _, _ = linalg.lstsq(A, s)
    sol = (np.round((sol)).astype(int) + sol_init) % n_period

    return sol


def roll_4D(data, shifts, in_place=False):
    """
    Roll a stack of periodic videos by the given shifts to obtain a 4D volume.
    This is a convenience wrapper around np.roll.

    Parameters
    ----------
    data : np.ndarray [T, Z, X, Y]
        The data to be synchronized.
    shifts : iterable (int)
        The computed shift of each layer
    in_place : bool, optional
        Overwrite the input array, by default False

    Returns
    -------
    np.ndarray [T, Z, X, Y]
        The shifted array.

    """

    if not in_place:
        data = data.copy()

    for n in range(1, data.shape[1]):
        data[:, n, ...] = np.roll(data[:, n, ...], -shifts[n], axis=0)

    return data
