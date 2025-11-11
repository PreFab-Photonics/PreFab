# PreFab

![PreFab logo](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/assets/prefab-logo.png?raw=true)

PreFab is a _virtual nanofabrication environment_ that leverages **deep learning** and **computer vision** to predict and correct for structural variations in integrated photonic devices during nanofabrication.

## Prediction

PreFab predicts process-induced structural variations, including corner rounding, loss of small lines and islands, filling of narrow holes and channels, sidewall angle deviations, and stochastic effects. This allows designers to rapidly prototype and evaluate expected performance pre-fabrication.

![Example of PreFab prediction](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/assets/promo_p.png?raw=true)

## Correction

PreFab corrects device designs to ensure that the fabricated outcome closely matches the intended specifications. This minimizes structural variations and reduces performance discrepancies between simulations and actual experiments.

![Example of PreFab correction](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/assets/promo_c.png?raw=true)

## Installation

Install PreFab using pip:

```sh
pip install prefab
```

For contributors who wish to make changes to the source code:

```sh
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the project
git clone https://github.com/PreFab-Photonics/PreFab.git
cd PreFab
uv sync
```

## Getting Started

Before you can make PreFab requests, you will need to [create an account](https://www.prefabphotonics.com/auth/signup).

To link your account, run the following command to authenticate your token:

```sh
prefab setup
```

Visit [`/docs/examples`](https://github.com/PreFab-Photonics/PreFab/tree/main/docs/examples) or our [documentation](https://docs.prefabphotonics.com/) to get started with your first predictions.

## License

LGPL-2.1 © PreFab Photonics
