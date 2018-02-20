# Progress
This document details some of the progress made on the library

## TODO
- [x] Add tests from DSDGP
- [x] Add method to initialize model params
- [x] Test DSDGP on toy example (add to docs)
- [ ] Test DSDGP and its optins further
- [ ] Work on improving documentation
- [ ] Look into multitask learning extensions

## Design Choices
* Sequential can be intialised with a partial list of layers, e.g. `[InputLayer,
  HiddenLayer, HiddenLayer]`. Should we disallow this in favour of the correct
layer structure, i.e. `OutputLayer` at the end?
* Currently `Sequential` only works by modifying `GPflow` source code. If pull
  request accepted nothing to be done here. Else, design needs to be changed.
* ~~`InputLayer` takes an initial value for Z, maybe better to disallow this and
  add it to the model object instead~~
* `Sequential` has an `_initialized` private attribute. Turn this into property.
* So far, multikernel layers have a set of inducing inputs for each kernel.
  One idea is to allow a shared set of inducing inputs for all kernels (this
could act as a form of regularasation)

## Known Bugs
* ~Prediction with full covariance doesn't work because of numerical issues~
* ~~Optimisation does not work~~
* Gpflow bug: methods decorated with `autoflow` do not take named arguments
