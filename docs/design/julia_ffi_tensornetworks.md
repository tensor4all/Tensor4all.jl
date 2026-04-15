# Julia Frontend TensorNetworks Layer

## Purpose

This document defines the public chain container that sits between the low-level
core primitives and the raw-array numerical layer.

## In Scope

- `TensorNetworks.TensorTrain`
- `TensorTrain = Vector{Tensor} + llim/rlim`
- `TensorNetworks.LinearOperator`
- `TensorNetworks.apply`
- space-binding helpers for `LinearOperator`
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

## `LinearOperator`

This layer owns the generic operator/application boundary.

```julia
mutable struct LinearOperator
    mpo::Union{TensorTrain, Nothing}
    input_indices::Vector{Index}
    output_indices::Vector{Index}
    true_input::Vector{Union{Index, Nothing}}
    true_output::Vector{Union{Index, Nothing}}
    metadata::NamedTuple
end
```

- In Phase 1, `metadata` carries the skeleton-level operator description.
- `mpo = nothing` is allowed until backend materialization is wired.
- `set_input_space!`, `set_output_space!`, and `set_iospaces!` belong here.
- `apply` also belongs here because it is generic over operator source.

## Boundary to SimpleTT

The `TensorNetworks` layer should not reimplement numerical TT kernels. When a
chain operation needs contraction or compression, the implementation should be
delegated to `SimpleTT` or to a reusable C API primitive.

## Remaining Helper Surface

The approved skeleton surface in this layer also includes:

- `findsite`
- `findsites`
- `findallsiteinds_by_tag`
- `findallsites_by_tag`
- `replace_siteinds!`
- `replace_siteinds`
- `replace_siteinds_part!`
- `rearrange_siteinds`
- `makesitediagonal`
- `extractdiagonal`
- `matchsiteinds`

In the current branch these names are part of the contract even where they still
throw `SkeletonNotImplemented`.

## HDF5 Boundary

`save_as_mps` and `load_tt` are the Julia-side persistence entry points for this
layer.

- save uses the `MPS` schema
- load returns `TensorNetworks.TensorTrain`
- the repository tests Tensor4all/ITensorMPS HDF5 interoperability in both
  directions
- the restored Julia docs assume a minimized chain-oriented C API target on the
  Rust side

## Open Questions

- How much chain-specific convenience should live in Julia versus Rust?
- Which operator-space binding helpers should move from skeleton to real
  implementation next?
