# Changelog

## [1.1.8](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.8) - 2025-01-16

- Added `prefab` command to the `pyproject.toml` file so that `prefab` can be run from the command line (e.g., `prefab setup`).
- Change to callback address for successful authentication.

## [1.1.7](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.7) - 2025-01-14

- Added `autograd` and `pydantic` (version requirement) dependencies in `pyproject.toml`.
- Server-side improvements (update endpoint version number to 2).

## [1.1.6](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.6) - 2024-12-30

- Added Tidy3D fabrication-aware inverse design (FAID) example notebook.
- Remove buffer from `Device` created with `read.from_sem` method.
- Handling of extended bounds in `read.from_sem` method.
- Use OpenCV for morphological operations in `geometry.enforce_feature_size`.
- Add handling for `None` casee for `BufferSpec` in `Device` constructor.

## [1.1.5](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.5) - 2024-11-05

- Fix alignment issue in `Device._device_to_gdstk` method, which is used in `Device.to_gdstk` and `Device.to_gds`.
- Minor linting fixes.

## [1.1.4](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.4) - 2024-11-01

- Added custom vector-Jacobian product (VJP) for the `predict.predict_array_with_grad` function.
- Changed some of the docstrings in `prefab.predict` to be more consistent and clear.

## [1.1.3](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.3) - 2024-10-26

- Moved prediction logic to `prefab.predict` module.
- First version of `predict.predict_array_with_grad`, which returns both the predicted array and its gradient. This is useful to fabrication-aware inverse design (FAID). More to come.
- Added `origin` parameter to GDS-related export methods.
- Small docstring fixes.

## [1.1.2](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.2) - 2024-10-10

- User warning if `compare.intersection_over_union`, `compare.hamming_distance`, or `compare.dice_coefficient` are called with non-binarized devices.
- Added `height` parameter to many shape constructors in `prefab.shapes` to give more flexibility.
- Updates to the `README.md` to keep current.
- `Device.is_binary` is now a property.
- Moved `Device.enforce_feature_size` logic to `prefab.geometry` module.
- Added required version of `gdstk` to `pyproject.toml`.
- Removed leftover return statement in `geometry.rotate`.

## [1.1.1](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.1.1) - 2024-09-24

- Manually adding small random noise to the "SEMulated" images to better match the real data. This is ideally included in the model training, but for now this is a quick fix.
- Added z-padding to the device array before exporting to STL with `Device.to_stl` to ensure that the exported device is closed.
- Removed buffer from `Device.device_array` before exporting to with `Device.to_gdsfactory`.
- The additions from `1.0.3` and `1.0.4` releases, which should be considered part of this release. Release planning a work in progress.
- Import and export from/to Tidy3D simulations with `Device.to_tidy3d` and `read.from_tidy3d`.
- Import and export from/to gdsfactory components with `Device.to_gdsfactory` and `read.from_gdsfactory`.
- Convert 2D device structures into 3D arrays or STL files with `Device.to_3d` and `Device.to_stl`. This is useful for simulating processes with angled sidewalls.
- Check and visualize the effect of enforcing a minimum feature size on the device geometry with `Device.check_feature_size` and `Device.enforce_feature_size`.

## [1.0.4](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.0.4) - 2024-09-19

- Option to specify GPU or CPU in `predict`, `correct`, and `semulate` functions. GPU option has more overhead and will take longer for small devices, but will be faster for larger devices.
- Improve clarity of messaging for authentication errors.

## [1.0.3](https://github.com/PreFab-Photonics/PreFab/releases/tag/v1.0.3) - 2024-09-14

- Added this `CHANGELOG.md` file.
- Added `prefab.shapes` module to replace the now removed `devices` directory. This module contains helpful device constructors for testing PreFab models on.
- Added `__version__` class attribute to `prefab` package.
- Added `ANT_NanoSOI_ANF1-d8` and `ANT_NanoSOI_ANF1-d10` models (see `prefab.models` and `docs/models.md`).
- Updated notebook examples to use `prefab.shapes` module and newest models.
- Updated dependencies in `pyproject.toml`.
- Simplified `.gitignore` and `.gitattributes` files.
- Docstring improvements to `prefab.compare` module.
- Ability to specify thickness to all four sides of buffer thickness in `prefab.device.BufferSpec`.
- Changed some of the array resizing logic in `prefab.read` module to be more robust.
- Minor fixes to `prefab.device` module.
- Minor fixes to `prefab.geometry` module.
- Removed `devices` directory from the repository. Effectively replaced with `prefab.shapes` module.
- Remove `requirements.txt` file as `pyproject.toml` contains all dependencies.
