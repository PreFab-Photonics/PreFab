---
title: Home
description: Documentation for PreFab - a virtual nanofab for photonic integrated circuit design
---

<img src="assets/prefab-logo-notext.svg" alt="PreFab logo" width="80">

Most photonic designs fail on the first fab run. PreFab eliminates costly design-fabrication iteration cycles by predicting manufacturing outcomes before tape-out. Using foundry-accurate models trained on real fabrication data, PreFab captures how lithography, etching, and process variations transform your designs during manufacturing.

## Prediction

Simulate exactly how your designs will fabricate. PreFab predicts the structural variations that matter: corner rounding, feature erosion, gap filling, sidewall angles, and stochastic effects. Compare ideal versus fabricated performance before committing to manufacturing.

![Example of PreFab prediction](assets/promo_p.png)

## Correction

Pre-compensate designs to counteract known fabrication effects. PreFab's correction models automatically adjust layouts so fabricated structures match your target specifications, recovering performance lost to manufacturing variations.

![Example of PreFab correction](assets/promo_c.png)

## Getting started

!!! tip "Quick links"

    1. [Setup guide](setup.md) 2. [Learn about fabrication-aware design](fab_aware_design.md) 3. [Explore example notebooks](examples/1_prediction.ipynb)

!!! info "Try Rosette (beta)"

    Want a more visual experience? Try the [Rosette beta](https://rosette.dev) - our new layout tool with PreFab built in, designed for rapid chip design.

## Get in touch

PreFab is continuously evolving to serve the photonics community. [Reach out](mailto:hi@prefabphotonics.com) to discuss support for your fabrication process, report issues on [GitHub](https://github.com/PreFab-Photonics/PreFab/issues), or follow updates on [LinkedIn](https://www.linkedin.com/company/prefab-photonics).
