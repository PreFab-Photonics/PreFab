# PreFab

![PreFab logo](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/assets/logo.png?raw=true)

PreFab is a _virtual nanofabrication environment_ that leverages **deep learning** and **computer vision** to predict and correct for structural variations in integrated photonic devices during nanofabrication.

> **Try Rosette**: Want a more visual experience? Try [Rosette](https://rosette.dev) - our new layout tool with PreFab models built in, designed for rapid chip design.

## Prediction

PreFab predicts process-induced structural variations, including corner rounding, loss of small lines and islands, filling of narrow holes and channels, sidewall angle deviations, and stochastic effects. This allows designers to rapidly prototype and evaluate expected performance pre-fabrication.

![Example of PreFab prediction](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/assets/promo_p.png?raw=true)

## Correction

PreFab corrects device designs to ensure that the fabricated outcome closely matches the intended specifications. This minimizes structural variations and reduces performance discrepancies between simulations and actual experiments.

![Example of PreFab correction](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/assets/promo_c.png?raw=true)

## Models

Each photonic nanofabrication process requires unique models, which are regularly updated with the latest data. The current models include (see the full list in [`docs/models.md`](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/models.md)):

| Foundry | Process | Latest Version    | Latest Dataset   | Model Name  |
| ------- | ------- | ----------------- | ---------------- | ----------- |
| ANT     | NanoSOI | ANF1 (May 6 2024) | d10 (Jun 8 2024) | ANT_NanoSOI_ANF1_d10 |
| ANT     | SiN     | ANF1 (May 6 2024) | d1 (Jan 31 2024) | ANT_SiN_ANF1_d1 |

> _New models are to be regularly added. Usage may change. For additional foundry and process models, feel free to [contact us](mailto:hi@prefabphotonics.com) or raise an issue._

## Installation

Install PreFab via pip:

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

If you have multiple Python environments and want to register PreFab as a Jupyter kernel:

```sh
python -m ipykernel install --user --name=prefab --display-name="PreFab"
```

### For Contributors

We recommend using [uv](https://docs.astral.sh/uv/) for development, which provides faster dependency resolution and better reproducibility.

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

## Getting Started

### Account setup

Before you can make PreFab requests, you will need to [create an account](https://www.prefabphotonics.com/login).

To link your account, you will need an token. You can do this by running the following command in your terminal. This will open a browser window where you can log in and authenticate your token.

```sh
prefab setup
```

### Guides

Visit [`/docs/examples`](https://github.com/PreFab-Photonics/PreFab/tree/main/docs/examples) or our [docs](https://docs.prefabphotonics.com/) to get started with your first predictions.

## Performance and Usage

PreFab models are hosted on a [serverless cloud platform](https://modal.com/). Please keep in mind:

- 🐢 Default CPU inference may be slower.
- 🥶 The first prediction using optional GPU inference may take longer due to cold start server loading. Subsequent predictions will be faster.
- 😊 Please be considerate of usage. Start with small tasks and limit usage during the initial stages. Thank you!

## License

This project is licensed under the LGPL-2.1 license. © 2025 PreFab Photonics.
