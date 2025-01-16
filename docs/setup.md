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

### From GitHub

For those who wish to make changes to the source code for their own development purposes, PreFab can also be installed directly from [GitHub](https://github.com/PreFab-Photonics/PreFab).

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
