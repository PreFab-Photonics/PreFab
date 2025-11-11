---
title: Setup
description: This guide will get you set up and ready to use the PreFab API.
---

Follow the steps below to install and authenticate the [PreFab Python package](https://pypi.org/project/prefab/).

!!! info "Python"

    If you are newer to Python, we recommend reading our [Python for Photonics](blog/posts/python-for-photonics.md) blog post.

## Install PreFab

### From PyPI (for users)

You can install PreFab using pip, which is the Python package installer. This method is suitable for most users.

```sh
pip install prefab
```

### From GitHub (for contributors)

For those who wish to contribute to PreFab or make changes to the source code, we recommend using [uv](https://docs.astral.sh/uv/) for development, which provides faster dependency resolution and better reproducibility.

**Using uv (recommended):**

```sh
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the project
git clone https://github.com/PreFab-Photonics/PreFab.git
cd PreFab
uv sync
```

**Using pip (alternative):**

```sh
git clone https://github.com/PreFab-Photonics/PreFab.git
cd PreFab
pip install -e .
```

## Sign up

Before you can make PreFab requests, you will need to create an account. Sign up [here](https://www.prefabphotonics.com/auth/signup).

## Authenticate PreFab token

To link your PreFab account to the API, you will need to create an authentication token. You can do this by running the following shell command. This will open a browser window where you can log in and generate a token.

```sh
prefab setup
```

## Verify installation

To verify that PreFab is setup correctly, you can run the following Python code.

```python
import prefab as pf

device = pf.shapes.target()
prediction = device.predict(model=pf.models["Generic_SOI"])
```

If the code runs without errors, you have successfully installed and authenticated PreFab. If not, please reach out to us at [support@prefabphotonics.com](mailto:support@prefabphotonics.com).
