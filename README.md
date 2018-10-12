# deep_generative

Provides a set of implementations for deep generative models. At the moment only [auto-encoding variational Bayes](https://arxiv.org/abs/1312.6114). This libarary offers the following utilities/classes that allow furthur expansion.

- Model class: for random variables (conditional or not) e.g. reparameterized distributions.
- Transform class: for neural network function and deterministic transformations.
- Normalizing flows: for measure preserving bijective transformation.
- Dynamics: for dynamical and time-correlated models.
