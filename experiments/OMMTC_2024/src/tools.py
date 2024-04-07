# Copyright (c) 2023 Idiap Research Institute, http://www.idiap.ch/
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
import scipy.interpolate


def compute_sampling_line(n_time, n_depth, slope, initial_phase, modulation, psf):
    t_domain = np.linspace(0, 1, n_time, endpoint=False)
    z_domain = np.linspace(0, 1, n_depth, endpoint=False)

    slope_coords = np.array(np.meshgrid(t_domain, z_domain, indexing="ij"))

    slope_theta = np.arctan(slope)
    sin = np.cos(slope_theta)
    cos = np.sin(slope_theta)

    rot_matrix = np.array([[cos, -sin], [sin, cos]])

    transformed_coords = np.einsum("nm,mtw->ntw", rot_matrix, slope_coords)

    elongation = np.sqrt(1 + slope**2) / slope

    modu_coords = np.linspace(0, 1, modulation.size, endpoint=False)
    modu_coords += modu_coords[1] / 2 * 0.99
    modu_interpolator = scipy.interpolate.RegularGridInterpolator(
        (modu_coords,),
        modulation,
        bounds_error=False,
        fill_value=None,
        method="nearest",
    )

    value_coords = np.linspace(
        0, 1, int(np.ceil(n_depth * elongation) * 2 + 1), endpoint=True
    )
    values = np.zeros((3, value_coords.size))
    values[1] = modu_interpolator(value_coords)
    value_coords *= elongation

    interpolator = scipy.interpolate.RegularGridInterpolator(
        ((-1 / n_time, 0, 1 / n_time), value_coords),
        values,
        bounds_error=False,
        fill_value=0,
        method="linear",
    )
    transformed_list = transformed_coords.reshape((2, -1)).T

    transformed_list[:, 0] -= initial_phase * cos
    transformed_list[:, 1] -= initial_phase * sin

    wrap_x = slope * sin
    wrap_y = np.sqrt(1 + slope**2) - slope * cos

    transformed_list[:, 1] += (transformed_list[1, 1] - transformed_list[0, 1]) / 2

    to_wrap = transformed_list[:, 0] < -wrap_x / 2
    while to_wrap.sum():
        transformed_list[:, 0][to_wrap] += wrap_x
        transformed_list[:, 1][to_wrap] += wrap_y
        to_wrap = transformed_list[:, 0] < -wrap_x / 2

    sampling_line = interpolator(transformed_list).T.reshape((n_time, n_depth))
    return scipy.ndimage.convolve1d(sampling_line, psf, axis=1, mode="constant")
