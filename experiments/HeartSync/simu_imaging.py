# Copyright (c) 2023 Idiap Research Institute, http://www.idiap.ch/
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

import argparse
import pathlib
import json

import numpy as np
import logging
from cbi_toolbox.simu import imaging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    path = pathlib.Path(args.config)
    print(path)

    with path.open("r") as fp:
        config = json.load(fp)

    path = path.parent

    logging.basicConfig(filename=str(path / "im_log.txt"), level=logging.INFO)

    data_path = path.parent / "data"
    if not data_path.exists():
        exc = FileNotFoundError(f"Data path does not exist: {str(data_path)}")
        logging.exception(exc)
        raise exc

    logging.info(f"Using config:\n{config}")

    if (path / "measure.npy").exists():
        logging.info("Images exist, skipping")
        exit(0)

    photons = config["photons"]
    dyn_range = config["dyn_range"]
    seed = config["seed"]

    sample = np.load(data_path / f"{config['sample']}.npy")
    psf = np.load(data_path / "psf.npy")
    illu = np.load(data_path / "spim.npy")

    measure = np.empty_like(sample)

    for frame, volume in enumerate(sample):
        logging.info(f"Imaging frame {frame}")
        measure[frame] = imaging.spim(volume, psf, illu)

    def simu_camera(image_in, dyn_range, photons, seed):
        image_in /= image_in.max()
        image_in *= dyn_range
        image_in = imaging.quantize(
            imaging.noise(image_in, photons=photons, seed=seed, clip=True, max_amp=1)
        )
        return image_in

    measure = simu_camera(measure, dyn_range, photons, seed)

    np.save(path / "measure.npy", measure)
