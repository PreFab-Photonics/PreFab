---
title: Setup
description: This guide will get you all set up and ready to use the PreFab API.
---

Follow the steps below to install and authenticate the [PreFab Python package](https://pypi.org/project/prefab/).

!!! info "Python"

    If you are new to Python, we recommend starting with the [Python for Photonics](blog/posts/python-for-photonics.md) blog post.

## Install PreFab

### From PyPI

You can easily install PreFab using pip, which is the Python package installer. This method is suitable for most users.

```sh
pip install prefab
```

PreFab includes `ipykernel` so you can use it directly in Jupyter notebooks, VS Code notebooks, and other notebook environments.

### Using in Jupyter Notebooks

After installation, PreFab is immediately available in your notebooks:

```python
import prefab as pf

# Create a device
device = pf.shapes.target()

# Make a prediction
prediction = device.predict(model=pf.models["ANT_NanoSOI"])

# Visualize
device.plot()
```

If you have multiple Python environments, you can register PreFab as a Jupyter kernel:

```sh
python -m ipykernel install --user --name=prefab --display-name="PreFab"
```

Then select the "PreFab" kernel from the kernel menu in Jupyter.

### From GitHub (For Contributors)

For those who wish to contribute to PreFab or make changes to the source code, we recommend using [uv](https://docs.astral.sh/uv/) for development, which provides faster dependency resolution and better reproducibility.

**Using uv (recommended):**

```sh
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the project
git clone https://github.com/PreFab-Photonics/PreFab.git
cd PreFab
uv sync

# Optional: Install Jupyter Lab if you want to launch it
uv sync --extra dev
uv run jupyter lab
```

**Using pip:**

```sh
git clone https://github.com/PreFab-Photonics/PreFab.git
cd PreFab
pip install -e .
```

## Sign up

Before you can make PreFab requests, you will need to create an account. Sign up [here](https://www.prefabphotonics.com/login). PreFab models are currently available for all registered users.

## Authenticate PreFab token

To link your PreFab account to the API, you will need to create an authentication token. You can do this by running the following command in your terminal. This will open a browser window where you can log in and generate a token.

```sh
prefab setup
```

## Verify installation

To verify that PreFab is setup correctly, you can run the following Python code.

```python
import prefab as pf

device = pf.shapes.target()
prediction = device.predict(model=pf.models["ANT_NanoSOI"])
```

If the code runs without errors, you have successfully installed and authenticated PreFab. If not, please reach out to us at [support@prefabphotonics.com](mailto:support@prefabphotonics.com).
