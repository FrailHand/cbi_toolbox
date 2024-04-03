"""
The reconstruct package provides reconstruction algorithms,
as well as preprocessing tools and performance scores.



[1] Liebling, M., et al. *"Four-dimensional cardiac imaging in living embryos via
postacquisition synchronization of nongated slice sequences."* Journal of
biomedical optics 10.5 (2005): 054001-054001
"""

# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>
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
import pywt
from cbi_toolbox import parallel as cbi_par


def psnr(ref, target, norm=None, limit=None, in_place=False):
    """
    Computes the Peak Signal-to-Noise Ratio:
    PSNR = 10 log( limit ^ 2 / MSE(ref, target) )

    Parameters
    ----------
    ref : numpy.ndarray
        The ground-truth reference array.
    target : numpy.ndarray
        The reconstructed array.
    norm : str
        Normalize the images before computing snr, default is None.
    limit: float, optional
        The maximum pixel value used for PSNR computation,
        default is None (max(ref)).
    in_place : bool, optional
        Perform normalizations in-place, by default False.

    Returns
    -------
    float
        The PSNR.
    """

    if norm is None:
        pass
    elif norm == "mse":
        target = scale_to_mse(ref, target, in_place)
    else:
        ref = normalize(ref, in_place)
        target = normalize(target, in_place)

    if limit is None:
        limit = ref.max()

    return 10 * np.log10(limit**2 / mse(ref, target))


def mse(ref, target):
    """
    Computes the Mean Squared Error between two arrays

    Parameters
    ----------
    ref : numpy.ndarray
        Reference array.
    target : numpy.ndarray
        Target array.

    Returns
    -------
    float
        The MSE.
    """

    return np.square(np.subtract(ref, target)).mean()


def normalize(image, mode="std", in_place=False):
    """
    Normalize an image according to the given criterion.

    Parameters
    ----------
    image : numpy.ndarray
        Image to normalize, will be modified.
    mode : str, optional
        Type of normalization to use, by default 'std'.
        Allowed: ['std', 'max', 'sum']
    in_place : bool, optional
        Perform computations in-place, by default False.

    Returns
    -------
    array
        The normalized image (same as input).

    Raises
    ------
    ValueError
        For unknown mode.
    """

    if mode == "std":
        f = np.std(image)
    elif mode == "max":
        f = np.max(image)
    elif mode == "sum":
        f = np.sum(image)
    else:
        raise ValueError("Invalid norm: {}".format(mode))

    if not in_place:
        image = image / f
    else:
        image /= f

    return image


def scale_to_mse(ref, target, in_place=False):
    """
    Scale a target array to minimise MSE with reference

    Parameters
    ----------
    ref : numpy.ndarray
        The reference for MSE.
    target : numpy.ndarray
        The array to rescale.
    in_place : bool, optional
        Perform computations in-place, by default False.

    Returns
    -------
    numpy.ndarray
        The rescaled target.
    """

    w = np.sum(ref * target) / np.sum(target**2)

    if not in_place:
        target = target * w
    else:
        target *= w

    return target


def mutual_information(sig_a, sig_b, bins=20):
    """
    Compute the mutual information between two signals.

    Parameters
    ----------
    sig_a : numpy.ndarray
        The first signal
    sig_b : numpy.ndarray
        The second signal
    bins : int, optional
        The number of bins used for probability density estimation, by default 20

    Returns
    -------
    float
        The mutual information
    """

    hist, _, _ = np.histogram2d(sig_a.ravel(), sig_b.ravel(), bins=bins)

    pxy = hist / float(np.sum(hist))
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    px_py = px[:, None] * py[None, :]

    nonzeros = pxy > 0

    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))


def dwt_preprocess(
    data,
    drop_hi=2,
    drop_lo=True,
    dist_threshold=0.85,
    level=None,
    wavelet="bior4.4",
    mode="periodic",
    parallel=False,
):
    """
    Preprocess the input images using wavelet transform to reduce their dimensionality
    and the noise.
    Inspired by the method described in [1].

    Parameters
    ----------
    data : np.ndarray [..., W, H]
        The input image(s) to process (the last 2 axes must correspond to the 2D
        input images)
    drop_hi : int, optional
        How many high-order wavelets to drop for size reduction, by default 2
    drop_lo : bool, optional
        Discard the lowest order wavelet to cancel the influence of global
        intensity shift, by default True
    dist_threshold : float, optional
        The threshold, by default 0.85
    level : int, optional
        Level of wavelet decomposition (see PyWavelet), by default None
    wavelet : str, optional
        Wavelet basis used (see PyWavelet), by default 'bior4.4'
    mode : str, optional
        Boundary conditions (see PyWavelet), by default 'periodic'
    parallel : bool, optional
        Run the transform in parallel over the first axis, by default False

    Returns
    -------
    np.ndarray
        The array of coefficients
    """
    dropped = 0
    axes = (-2, -1)

    if parallel:
        n_axes = data.ndim - 2
        wave = pywt.wavedec2(
            data[(0,) * n_axes], wavelet=wavelet, mode=mode, level=level, axes=axes
        )

        if drop_hi > 0:
            wave = wave[:-drop_hi]
        if drop_lo:
            wave[0] *= 0
            dropped = wave[0].size * np.prod(data.shape[:n_axes])

        wave, _ = pywt.coeffs_to_array(wave, axes=axes)

        wave = np.empty_like(wave, shape=(*data.shape[:n_axes], *wave.shape))

        def process_inner(start, width):
            wave_in = pywt.wavedec2(
                data[start : start + width],
                wavelet=wavelet,
                mode=mode,
                level=level,
                axes=axes,
            )

            if drop_hi > 0:
                wave_in = wave_in[:-drop_hi]
            if drop_lo:
                wave_in[0] *= 0

            wave[start : start + width], _ = pywt.coeffs_to_array(wave_in, axes=axes)

        cbi_par.parallelize(process_inner, data.shape[0])

    else:
        wave = pywt.wavedec2(data, wavelet=wavelet, mode=mode, level=level, axes=axes)

        if drop_hi > 0:
            wave = wave[:-drop_hi]
        if drop_lo:
            wave[0] *= 0
            dropped = wave[0].size

        wave, _ = pywt.coeffs_to_array(wave, axes=axes)

    if dist_threshold > 0:
        # First rough histogram to ckeck which region to focus on
        hist, bins = np.histogram(np.abs(wave), bins=100)
        hist[0] -= dropped
        hist = np.cumsum(hist) / wave.size

        # Check the range needed to include the desired threshold value
        rmax = bins[np.searchsorted(hist, dist_threshold) + 1]

        # Compute only the focused histogram
        hist, bins = np.histogram(np.abs(wave), bins=100, range=(0, rmax))
        hist[0] -= dropped
        hist = np.cumsum(hist) / (wave.size)

        threshold = bins[np.searchsorted(hist, dist_threshold)]

        wave = pywt.threshold(wave, threshold)

    return wave
