# PreFab

![PreFab logo](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/logo.png?raw=true)

`PreFab` models fabrication process induced structural variations in integrated photonic devices using **deep learning**. New insights into the capabilities of nanofabrication processes are uncovered and device design fidelity is enhanced in this *virtual nanofabrication environment*.

## Prediction

`PreFab` predicts process-induced structural variations such as corner rounding (both over and under etching), washing away of small lines and islands, and filling of narrow holes and channels in planar photonic devices. The designer then resimulates their (predicted) design to rapidly prototype the expected performance and make any necessary corrections prior to nanofabrication.

![Example of PreFab prediction](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/promo_p.png?raw=true) *Predicted fabrication variation of a star structure on a silicon-on-insulator e-beam lithography process.*

## Correction

`PreFab` also makes automatic corrections to device designs so that the fabricated outcome is closer to the nominal design. Less structural variation generally means less performance degradation from simulation to experiment.

![Example of PreFab correction](https://github.com/PreFab-Photonics/PreFab/blob/main/assets/promo_c.png?raw=true) *Corrected fabrication of a star structure on a silicon-on-insulator e-beam lithography process.*

## Models

Each photonic foundry requires its own *predictor* and *corrector* models. These models are updated regularly based on data from recent fabrication runs. The following models are currently available:

| Foundry | Process | Latest Version (Date) | Type | Status | Name | Usage|
| --- | ------- | --------------- | --------- | ----- | ---------------- | ---|
| ANT | [NanoSOI](https://www.appliednt.com/nanosoi-fabrication-service/) | v2 (Jan 9 2023) | Predictor | Beta | p_ANT_NanoSOI_v2 | Open |
| ANT | [NanoSOI](https://www.appliednt.com/nanosoi-fabrication-service/) | v2 (Jan 9 2023) | Corrector | Beta | c_ANT_NanoSOI_v2 | Open |

*This list will update over time. Usage is subject to change. Please contact us or create an issue if you would like to see a new foundry and process modelled.*

## Installation

### Local

Install the latest version of `PreFab` locally using:

```sh
pip install prefab
```

### Online

Run the latest version of `PreFab` online by following:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new?machine=basicLinux32gb&repo=608330448&ref=main&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=EastUs)

## Getting Started

Please see the notebooks in `/examples` to get started with making predictions.

## Performance and Usage

Currently, `PreFab` models are accessed through a serverless cloud platform that has the following limitations to keep in mind:

- Inferencing is done on a CPU and will thus be relatively slow. GPU inferencing to come in future updates.
- The first prediction may be slow, as the cloud server may have to perform a cold start and load the necessary model(s). After the first prediction, the server will be "hot" for some time and the subsequent predictions will be much quicker.
- Please be mindful of your usage. Start with small examples before scaling up to larger device designs, and please keep usage down to a reasonable amount in these early stages. Thank you!

<!-- ## Documentation

To get started with tutorials and examples, or to dive into the API reference, visit our full documentation [here](README.md). -->

## License

This project is licensed under the terms of the LGPL-2.1 license. © 2023 PreFab Photonics.
