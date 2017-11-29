This document covers planned features for this library. More will be added
later.

# Modularity
* Implementation of Keras like API for DGPs; for instance, initialising each
  layer on its own and then stacking the layers as a network.
* Seperatin of model and inference procedure (similar to what GPflow does)

# Multitask/Multi-kernel Support
* Ability to create layers with multiple kernels
* Ability to perform multi-task learning

# Inference Procedures
* Doubly Stochastic VI (Salimbeni and Diesenroth 2017)
* Standard VI (Damianou and Lawrence 2013)
* Some for of MCMC

