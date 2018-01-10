# Progress
This document details some of the progress made on the library

## TODO

## Design Choices
* Sequential can be intialised with a partial list of layers, e.g. `[InputLayer,
  HiddenLayer, HiddenLayer]`. Should we disallow this in favour of the correct
layer structure, i.e. `OutputLayer` at the end?
* Currently `Sequential` only works by modifying `GPflow` source code. If pull
  request accepted nothing to be done here. Else, design needs to be changed.