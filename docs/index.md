---
title: Installation
description: This guide will get you all set up and ready to use the PreFab API.
---

PreFab leverages deep learning to model fabrication-induced structural variations in integrated photonic devices. Through this _virtual nanofabrication environment_, we uncover valuable insights into nanofabrication processes and enhance device design accuracy.

This guide will get you all set up and ready to use PreFab to predict the fabrication-induced structural variations of your own integrated photonic devices.

!!! note "Sign up"

    Before you can make PreFab requests, you will need to create an account[^1]. Sign
    up [here](https://www.prefabphotonics.com/signup).

## Install PreFab

Before making your first prediction, follow the steps below to install the [PreFab Python package](https://pypi.org/project/prefab/).

### From PyPI

You can easily install PreFab using pip, which is the Python package installer. This method is suitable for most users.

```sh
pip install prefab
```

### From GitHub

For those who wish to make changes to the source code for their own development purposes, PreFab can also be installed directly from [GitHub](https://github.com/PreFab-Photonics/PreFab).

```sh
git clone https://github.com/PreFab-Photonics/PreFab.git
pip install -e prefab
```

## Authenticate PreFab token

To link your PreFab account to the API, you will need to create an authentication token. You can do this by running the following command in your terminal. This will open a browser window where you can log in and generate a token.

```sh
python3 -m prefab setup
```

## PreFab~ricate~ your designs

See the following guides to get started with making your first predictions and corrections of fabrication-induced variations with PreFab:

1. [Making a prediction](examples/1_prediction.ipynb)
2. [Making a correction](examples/2_correction.ipynb)

!!! tip "Performance and usage"

    PreFab models are served via a serverless [cloud platform](https://modal.com/). Please note:

    - 🐢 CPU inference may result in slower performance. Future updates will introduce GPU inference.
    - 🥶 The first prediction may take longer due to cold start server loading. Subsequent predictions will be faster.
    - 😊 Be considerate of usage. Start small and limit usage during the initial stages. Thank you!

## Your thoughts are valuable

PreFab is a new design tool, still in its early days, that we hope will become useful to the photonics community. We are eager to hear about your experiences with PreFab. Please share your thoughts [with us](mailto:dusan@prefabphotonics.com) and any issues you may have on [GitHub](https://github.com/PreFab-Photonics/PreFab/issues).

Happy designing :fontawesome-solid-computer: :material-arrow-right: :material-chip:

[^1]: For more information, visit our [Privacy Policy](https://www.prefabphotonics.com/legal/privacy-policy) and [Terms of Service](https://www.prefabphotonics.com/legal/terms). <!-- markdownlint-disable-line MD053 -->
