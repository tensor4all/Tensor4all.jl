# Tensor4all.jl

> Current phase: restored old Julia frontend POC.

`Tensor4all.jl` is currently organized around the older Julia-facing module
split rather than the TreeTN-first skeleton.

## Primary Public Modules

- `TensorNetworks` for indexed chain objects
- `SimpleTT` for raw-array tensor trains
- `TensorCI` for cross interpolation that returns `SimpleTT`
- `QuanticsTransform` for Julia-owned operator semantics
- adopted `QuanticsGrids.jl` re-export for grid semantics

## Current Public Story

- `TensorNetworks.TensorTrain` is the indexed chain type and stores
  `Vector{Tensor}` plus `llim` / `rlim`.
- `SimpleTT.TensorTrain{T,N}` owns raw-array compression and MPO contraction.
- `TensorCI.crossinterpolate2` returns `SimpleTT.TensorTrain`.
- HDF5 interoperability is provided in pure Julia through `save_as_mps` and
  `load_tt`.

## Secondary / Deferred Surface

`TreeTensorNetwork` and other TreeTN-oriented skeletons may still exist in the
codebase, but they are not the primary public architecture of this branch.

## Entry Points

- [Module Overview](modules.md)
- [API Notes](api.md)
- [Design Documents](design_documents.md)
- [Deferred Rework Plan](deferred_rework_plan.md)
