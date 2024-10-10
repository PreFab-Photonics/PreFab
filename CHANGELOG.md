# Changelog

## 1.1.2 -

### Added

- User warning if `compare.intersection_over_union`, `compare.hamming_distance`, or `compare.dice_coefficient` are called with non-binarized devices.

### Changed

- Updates to the `README.md` to keep current.
- `Device.is_binary` is now a property.

## 1.1.1 - 2024-09-24

### Changed

- Manually adding small random noise to the semulated images to better match the real data. This is ideally included in the model training, but for now this is a quick fix.
- Added z-padding to the device array before exporting to STL with `Device.to_stl` to ensure that the exported device is closed.

### Fixed

- Removed buffer from `Device.device_array` before exporting to with `Device.to_gdsfactory`.

### Added

- The additions from `1.0.3` and `1.0.4` releases, which should be considered part of this release. Release planning a work in progress.
- Import and export from/to Tidy3D simulations with `Device.to_tidy3d` and `read.from_tidy3d`.
- Import and export from/to gdsfactory components with `Device.to_gdsfactory` and `read.from_gdsfactory`.
- Convert 2D device structures into 3D arrays or STL files with `Device.to_3d` and `Device.to_stl`. This is useful for simulating processes with angled sidewalls.
- Check and visualize the effect of enforcing a minimum feature size on the device geometry with `Device.check_feature_size` and `Device.enforce_feature_size`.

## 1.0.4 - 2024-09-19

### Added

- Option to specify GPU or CPU in `predict`, `correct`, and `semulate` functions. GPU option has more overhead and will take longer for small devices, but will be faster for larger devices.

### Changed

- Improve clarity of messaging for authentication errors.

## 1.0.3 - 2024-09-14

### Added

- Added this `CHANGELOG.md` file.
- Added `prefab.shapes` module to replace the now removed `devices` directory. This module contains helpful device constructors for testing PreFab models on.
- Added `__version__` class attribute to `prefab` package.

### Changed

- Added `ANT_NanoSOI_ANF1-d8` and `ANT_NanoSOI_ANF1-d10` models (see `prefab.models` and `docs/models.md`).
- Updated notebook examples to use `prefab.shapes` module and newest models.
- Updated dependencies in `pyproject.toml`.
- Simplified `.gitignore` and `.gitattributes` files.
- Docstring improvements to `prefab.compare` module.
- Ability to specify thickness to all four sides of buffer thickness in `prefab.device.BufferSpec`.
- Changed some of the array resizing logic in `prefab.read` module to be more robust.

### Fixed

- Minor fixes to `prefab.device` module.
- Minor fixes to `prefab.geometry` module.

### Removed

- Removed `devices` directory from the repository. Effectively replaced with `prefab.shapes` module.
- Remove `requirements.txt` file as `pyproject.toml` contains all dependencies.
