# PreFab

![PreFab logo](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/logo.png?raw=true)

`PreFab` leverages **deep learning** to model fabrication-induced structural variations in integrated photonic devices. Through this *virtual nanofabrication environment*, we uncover valuable insights into nanofabrication processes and enhance device design accuracy.

## Prediction

`PreFab` accurately predicts process-induced structural alterations such as corner rounding, washing away of small lines and islands, and filling of narrow holes in planar photonic devices. This enables designers to quickly prototype expected performance and rectify designs prior to nanofabrication.

![Example of PreFab prediction](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/promo_p.png?raw=true)

## Correction

`PreFab` automates corrections to device designs, ensuring the fabricated outcome aligns with the original design. This results in reduced structural variation and performance disparity from simulation to experiment.

![Example of PreFab correction](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/promo_c.png?raw=true)

## Models

`PreFab` accommodates unique *predictor* and *corrector* models for each photonic foundry, regularly updated based on recent fabrication data. Current models include (see full list on [`docs/models.md`](docs/models.md)):

| Foundry | Process | Latest Version | Latest Dataset | Type | Full Name | Status | Usage |
| ------- | ------- | -------------- | -------------- | ---- | --------- |------- | ----- |
| ANT | [NanoSOI](https://www.appliednt.com/nanosoi-fabrication-service/) | v5 (Jun 3 2023) | v4 (Apr 12 2023) | Predictor | p_ANT_NanoSOI_v5_d4 | Beta | Open |
| ANT | [NanoSOI](https://www.appliednt.com/nanosoi-fabrication-service/) | v5 (Jun 3 2023) | v4 (Apr 12 2023) | Corrector | c_ANT_NanoSOI_v5_d4 | Beta | Open |

*New models and foundries are regularly added. Usage may change. For additional foundry and process models, feel free to contact us or raise an issue.*

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

Visit [`/examples`](https://github.com/PreFab-Photonics/PreFab/tree/main/examples) for usage notebooks.

## Performance and Usage

`PreFab` models are served via a serverless cloud platform. Please note:

- 🐢 CPU inferencing may result in slower performance. Future updates will introduce GPU inferencing.
- 🥶 The first prediction may take longer due to cold start server loading. Subsequent predictions will be faster.
- 😊 Be considerate of usage. Start small and limit usage during the initial stages. Thank you!

## License

This project is licensed under the LGPL-2.1 license. © 2023 PreFab Photonics.
