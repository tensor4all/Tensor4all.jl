# Julia Frontend TensorNetworks Layer

## Purpose

This document defines the public chain container that sits between the low-level
core primitives and the raw-array numerical layer.

## In Scope

- `TensorNetworks.TensorTrain`
- `TensorTrain = Vector{Tensor} + llim/rlim`
- runtime topology checks for chain-only operations
- the HDF5 boundary that stores and loads chain data
- the fact that `MPS` and `MPO` are runtime conventions, not separate Julia
  types

This layer does not own raw-array TT compression or interpolation kernels. Those
remain in `SimpleTT` and `TensorCI`.

## `TensorTrain`

```julia
mutable struct TensorTrain
    data::Vector{Tensor}
    llim::Int
    rlim::Int
end
```

- `data` stores the site tensors in chain order.
- `llim` and `rlim` define the active chain bounds.
- `length(tt)` is the number of site tensors.
- `MPS` and `MPO` are public conventions over the same container.

## Runtime Checks

Operations that need chain structure must validate the topology at runtime.
Examples include:

- chain shape
- MPS-like site-index count
- MPO-like site-index count

This layer should raise descriptive Julia errors when the structure is not
compatible.

## Boundary to SimpleTT

The `TensorNetworks` layer should not reimplement numerical TT kernels. When a
chain operation needs contraction or compression, the implementation should be
delegated to `SimpleTT` or to a reusable C API primitive.

## HDF5 Boundary

`save_as_mps` and `load_tt` are the Julia-side persistence entry points for this
layer.

- save uses the `MPS` schema
- load returns `TensorNetworks.TensorTrain`
- the restored Julia docs assume a minimized chain-oriented C API target on the
  Rust side

## Open Questions

- How much chain-specific convenience should live in Julia versus Rust?
- Which operations should remain wrappers around `Tensor` primitives, and which
  ones should be delegated to `SimpleTT`?
