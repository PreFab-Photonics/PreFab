---
title: Fab-aware design
description: Designing high-performance photonic circuits with fabrication-aware modeling.
---

PreFab bridges the gap between photonic design and fabrication by incorporating real fabrication effects directly into the design process. Deep learning and computer vision are used to predict and compensate for manufacturing variations that traditional approaches often ignore due to lack of useful data. This enables accurate prediction, verification, and optimization of designs with fabrication effects in mind to create manufacturable designs that are as close to the theoretical limit as possible, while also reducing the need for manual design iteration and costly repeated fabrication runs.

!!! note "Note"

    While our general documentation focuses on planar geometric variations, the same principles extend to other critical fabrication effects. PreFab supports or is developing support for other effects such as sidewall angle, material thickness variations, and surface roughness - all of which significantly impact photonic device performance.

In this document, we'll explore four key aspects of fabrication-aware design:

1. [Modelling fabrication awareness](#modelling-fabrication-awareness)
2. [Design verification](#design-verification)
3. [Design correction](#design-correction)
4. [Fabrication-aware inverse design](#fabrication-aware-inverse-design)

## Modelling fabrication awareness

```mermaid
flowchart LR
    A[GDS<br>Cells] --> B[Prediction<br>Model]
    A --> F[Nanofabrication<br>and Imaging]
    F --> E
    B --> C[Predicted<br>Cells]
    C --> D[Calculate<br>Differences]
    D --> |Training| B
    E[SEM<br>Images] --> D
    style A fill:none,color:white,stroke:white
    style B fill:none,color:white,stroke:lightgreen
    style C fill:none,color:white,stroke:white
    style D fill:none,color:white,stroke:white
    style E fill:none,color:white,stroke:white
    style F fill:none,color:white,stroke:lightblue
    linkStyle default stroke:white
```

The foundation of fabrication-aware design lies in building a model that accurately predicts how designs will manifest on the chip. Our modeling system achieves this through a pipeline that combines deep learning with real nanofabrication data, as shown in the diagram above.

!!! tip "Models"

    PreFab handles this part of the process. See our [models documentation](models.md) for more details. [Reach out to us](mailto:hi@prefabphotonics.com) if you'd like to see support for a new fabrication process.

A nanofabrication process transforms designs into physical devices through multiple steps, each with unique parameters and inherent variability across fabrication runs and wafer locations. Our approach simplifies this complexity by directly modeling the relationship between design inputs and fabricated outcomes, without modeling intermediate steps.

The foundation of this system is our training data generation. GDS layout cells with varying features and distributions are fabricated, followed by high-resolution SEM imaging to capture the results. This creates paired design-outcome datasets that represent the complete process behavior.

At the heart of the system, a neural network model processes GDS layouts to predict post-fabrication structures. The model continuously refines its predictions by comparing against SEM images and updating its understanding through iterative training. Through this process, we train separate models for each fabrication process, enabling accurate predictions of manufacturing outcomes.

> For a simple look at how the prediction model works for a single process, see our [prediction notebook](examples/1_prediction.ipynb).

## Design verification

```mermaid
flowchart LR
    A[Device<br>Design] --> B[Prediction<br>Model]
    B --> C[Predicted<br>Design]
    C --> D[EM<br>Simulation]
    D --> E[Compare<br>Results]
    A --> D
    E -.-> |Manual Design Iteration| A
    style A fill:none,color:white,stroke:white
    style B fill:none,color:white,stroke:lightgreen
    style C fill:none,color:white,stroke:white
    style D fill:none,color:white,stroke:yellow
    style E fill:none,color:white,stroke:white
    linkStyle default stroke:white
```

Once we can predict fabrication outcomes with a prediction model, the next step is verifying how they'll affect device performance. The verification workflow combines these predictions with electromagnetic simulations like FDTD to provide a better picture of real-world behavior.

The process begins with a prediction of how the device design will appear after fabrication. The original and predicted structures are simulated to understand how fabrication effects will impact performance. Especially for complex devices that push the limits of fabrication, this comparison provides valuable insights for design iteration, enabling data-driven decisions about necessary adjustments.

This approach eliminates much of the guesswork in photonic circuit development. Rather than relying on intuition or empirical rules of thumb, designers can make informed choices based on accurate predictions of fabricated performance. By PreFab being written in Python, we can easily integrate with other simulation and layout tools.

> For a practical example of this workflow, see our [prediction and simulation tutorial](examples/5_prediction_simulation.ipynb).

## Design correction

```mermaid
flowchart LR
    A[Device<br>Design] --> B[Correction<br>Model]
    B --> C[Corrected<br>Design]
    C -.-> D[Design<br>Verification]
    style A fill:none,color:white,stroke:white
    style B fill:none,color:white,stroke:lightgreen
    style C fill:none,color:white,stroke:white
    style D fill:none,color:white
    linkStyle default stroke:white
```

While verification helps identify potential issues, design correction takes this a step further by automatically compensating for fabrication effects. The correction model works similarly to the prediction model but in reverse—instead of predicting fabricated shapes from designs, it generates designs that will fabricate into desired shapes.

The correction model generates a corrected design with precise geometric adjustments—adding material where erosion is expected, removing it where dilation is anticipated, and adjusting corners to account for rounding effects. This corrected design can be sent directly to fabrication or, optionally, through the verification workflow for additional confidence. As many designs—even simple ones—can create impossible-to-replicate structures (e.g., a 90° corner), the correction process will never be able to create a perfect design (we are still bound to fundamental physical limits), but we find that corrections always offer a significant degree of useful improvement.

This approach transforms fabrication effects from an unavoidable source of error into a controllable variable. Instead of using conservative design margins, we enable targeted corrections that push designs closer to their theoretical limits while maintaining manufacturability.

For existing designs, this correction workflow is a great way to improve performance without having to start from scratch.

> For a simplified example of design correction, see our [correction notebook](examples/2_correction.ipynb).

## Fabrication-aware inverse design

```mermaid
flowchart LR
    A[Device<br>Design] --> B[Prediction<br>Model]
    B --> C[Predicted<br>Design]
    C --> D[EM<br>Simulation]
    D --> E[Gradient<br>Calculation]
    E --> |Optimization| A
    style A fill:none,color:white,stroke:white
    style B fill:none,color:white,stroke:lightgreen
    style C fill:none,color:white,stroke:white
    style D fill:none,color:white,stroke:yellow
    style E fill:none,color:white,stroke:white
    linkStyle default stroke:white
```

The ultimate integration of fabrication awareness comes in the form of fabrication-aware inverse design or FAID, where fabrication effects are incorporated directly into the optimization process itself. Rather than appending fabrication awareness as a post-processing step, here it is incorporated into the optimization loop itself to create designs that are inherently manufacturable.

This is especially useful in designing around aspects of the fabrication process (e.g., etching and stochastic effects) that cannot be corrected with design parameter tuning. **In other words, FAID produces the best possible design given the limits of fabrication.**

The optimization loop begins with an initial design and uses the prediction model to anticipate its fabricated form. This predicted structure is then simulated, from which we calculate gradients that guide design updates (depending on the optimization algorithm). The process repeats, continuously refining the design while accounting for both performance targets and fabrication constraints. The result is designs that are inherently manufacturable while meeting performance requirements.

> For a detailed example of this advanced workflow, see our [fabrication-aware inverse design tutorial](examples/6_fabrication-aware_inverse_design.ipynb).
