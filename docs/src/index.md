# Tensor4all.jl

> Current phase: implementation of the restored Julia frontend.

`Tensor4all.jl` is organized around the older Julia-facing module split rather
than the removed TreeTN-first layout.

## Primary Public Modules

- `TensorNetworks` for indexed chain objects
- `SimpleTT` for raw-array tensor trains
- `TensorCI` for cross interpolation that returns `TensorCI2`
- `QuanticsGrids` as the adopted grid re-export layer
- `QuanticsTCI` as the adopted quantics-TCI re-export layer
- `QuanticsTransform` for quantics-specific operator constructors

## Current Public Story

- `TensorNetworks.TensorTrain` is the indexed chain type and stores
  `Vector{Tensor}` plus `llim` / `rlim`.
- `TensorNetworks.LinearOperator` and `TensorNetworks.apply` are the generic
  operator/application boundary.
- `SimpleTT.TensorTrain{T,N}` owns raw-array compression and MPO contraction.
- `TensorCI.crossinterpolate2` returns `TensorCI2` for the supported multi-site
  path.
- `SimpleTT.TensorTrain(tci)` converts interpolation results into raw-array TT
  form.
- HDF5 interoperability is provided in pure Julia through `save_as_mps` and
  `load_tt`.

## Still Missing

The current branch is intentionally chain-oriented. Broader non-chain work,
deeper backend integration, and some `QuanticsTransform` kernels are still not
implemented.

## Entry Points

- [Getting Started](getting_started.md) — tutorial with code examples
- [Module Overview](modules.md)
- [API Notes](api.md)
- [Design Documents](design_documents.md)
- [Deferred Rework Plan](deferred_rework_plan.md)
