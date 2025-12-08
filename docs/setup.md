---
title: Setup
description: This guide will get you set up and ready to use the PreFab API.
---

Follow the steps below to install and authenticate the [PreFab Python package](https://pypi.org/project/prefab/).

!!! info "Python"

    If you are newer to Python, we recommend reading our [Python for Photonics](python-for-photonics.md) guide.

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

## Authenticate PreFab token (easiest)

To link your PreFab account to the API, you will need to create an authentication token. You can do this by running the following shell command. This will open a browser window where you can log in and generate a token.

```sh
prefab setup
```

## Remote server authentication (alternative)

For remote servers, CI/CD pipelines, or environments without browser access, you can use an API key instead of the browser-based setup above.

### Generate an API key

1. Log in to your [account dashboard](https://www.prefabphotonics.com/settings/profile)
2. Find the **API Keys** section
3. Click **Generate** and copy the key (it will only be shown once)

### Set the environment variable

Set the `PREFAB_API_KEY` environment variable with your key:

```sh
export PREFAB_API_KEY=<your_key_here>
```

For persistent configuration, add the above line to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.) or your deployment environment's secrets configuration.

## Verify installation

To verify that PreFab is setup correctly, you can run the following Python code.

```python
import prefab as pf

device = pf.shapes.target()
prediction = device.predict(model=pf.models["Generic_SOI"])
```

If the code runs without errors, you have successfully installed and authenticated PreFab. If not, please reach out to us at [support@prefabphotonics.com](mailto:support@prefabphotonics.com).
