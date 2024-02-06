# PreFab

![PreFab logo](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/logo.png?raw=true)

`PreFab` leverages **deep learning** to model fabrication-induced structural variations in integrated photonic devices. Through this _virtual nanofabrication environment_, we uncover valuable insights into nanofabrication processes and enhance device design accuracy.

## Prediction

`PreFab` accurately predicts process-induced structural alterations such as corner rounding, washing away of small lines and islands, and filling of narrow holes in planar photonic devices. This enables designers to quickly prototype expected performance and rectify designs prior to nanofabrication.

![Example of PreFab prediction](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/promo_p.png?raw=true)

## Correction

`PreFab` automates corrections to device designs, ensuring the fabricated outcome aligns with the original design. This results in reduced structural variation and performance disparity from simulation to experiment.

![Example of PreFab correction](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/promo_c.png?raw=true)

## Models

`PreFab` accommodates unique _predictor_ and _corrector_ models for each photonic foundry, regularly updated based on recent fabrication data. Current models include (see full list on [`docs/models.md`](https://github.com/PreFab-Photonics/PreFab/blob/main/docs/models.md)):

| Foundry   | Process                                                                            | Latest Version   | Latest Dataset   | Model Name    | Model Tag   | Status |
| --------- | ---------------------------------------------------------------------------------- | ---------------- | ---------------- | ------------- | ----------- | ------ |
| ANT       | [NanoSOI](https://www.appliednt.com/nanosoi-fabrication-service/)                  | v6 (Nov 24 2023) | d7 (Nov 24 2023) | ANT_NanoSOI   | v6-d7       | Beta   |
| ANT       | [SiN (Upper Edge)](https://www.appliednt.com/nanosoi/sys/resources/specs_nitride/) | v5 (Jun 3 2023)  | d0 (Jun 1 2023)  | ANT_SiN       | v5-d0-upper | Alpha  |
| ANT       | [SiN (Lower Edge)](https://www.appliednt.com/nanosoi/sys/resources/specs_nitride/) | v5 (Jun 3 2023)  | d0 (Jun 1 2023)  | ANT_SiN       | v5-d0-lower | Alpha  |
| SiEPICfab | [SOI](https://siepic.ca/fabrication/)                                              | v5 (Jun 3 2023)  | d0 (Jun 14 2023) | SiEPICfab_SOI | v5-d0       | Alpha  |

_New models and foundries are to be added. Usage may change. For additional foundry and process models, feel free to contact us or raise an issue._

## Installation

### Local

Install `PreFab` via pip:

```sh
pip install prefab
```

Or clone the repository and install in development mode:

```sh
git clone https://github.com/PreFab-Photonics/PreFab.git
cd PreFab
pip install -e .
```

### Online

Use `PreFab` online through GitHub Codespaces:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?machine=basicLinux32gb&repo=608330448&ref=main&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=EastUs)

## Getting Started

### Account setup

Before you can make PreFab requests, you will need to [create an account](https://www.prefabphotonics.com/login).

To link your account, you will need a token. You can do this by running the following command in your terminal. This will open a browser window where you can log in and authenticate your token.

```sh
python3 -m prefab setup
```

### Guides

Visit [`/examples`](https://github.com/PreFab-Photonics/PreFab/tree/main/examples) or our [Guides](https://www.prefabphotonics.com/docs/guides/predictions) to get started with your first predictions.

## Performance and Usage

`PreFab` models are served via a serverless cloud platform. Please note:

- 🐢 CPU inference may result in slower performance. Future updates will introduce GPU inference.
- 🥶 The first prediction may take longer due to cold start server loading. Subsequent predictions will be faster.
- 😊 Be considerate of usage. Start small and limit usage during the initial stages. Thank you!

## License

This project is licensed under the LGPL-2.1 license. © 2024 PreFab Photonics.
