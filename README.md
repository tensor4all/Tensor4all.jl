# Tensor4all.jl

Julia frontend for [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs).

## Current Direction

This branch restores the older Julia-facing architecture:

- `TensorNetworks.TensorTrain` is the indexed chain type
- `SimpleTT.TensorTrain{T,N}` is the raw-array tensor-train layer
- `TensorCI` returns `SimpleTT`
- `QuanticsTransform` is a Julia-owned operator layer
- HDF5 roundtrip is handled directly through `HDF5.jl`

`TreeTensorNetwork` still exists in the repository, but it is not the primary
public story of this branch.

## What Works In This POC

- pure Julia `Index` and `Tensor` metadata types
- `TensorNetworks.TensorTrain` skeleton with `data`, `llim`, and `rlim`
- `SimpleTT` compression with `:LU`, `:CI`, and `:SVD`
- pure Julia MPO-MPO contraction for `SimpleTT` with `:naive` and `:zipup`
- `TensorCI.crossinterpolate2` returning `SimpleTT.TensorTrain`
- HDF5.jl-backed MPS-schema roundtrip through `save_as_mps` / `load_tt`
- adopted quantics grid re-exports from `QuanticsGrids.jl`

## What Is Still Deferred

- full docs cleanup beyond the restored architecture story
- finalized Julia-facing reduced `tensor4all-rs` ABI documentation
- deeper `QuanticsTransform` kernels and wrappers
- broader `TreeTensorNetwork` / non-chain functionality

## Development Notes

- `tensor4all-rs` remains the backend for low-level kernels.
- `Tensor4all.jl` owns the pure Julia public object model.
- `TensorCrossInterpolation.jl` is used as an implementation dependency for the
  current `SimpleTT` / `TensorCI` POC.

## Build Script

The backend build script looks for `tensor4all-rs` in this order:

1. `TENSOR4ALL_RS_PATH`
2. sibling directory `../tensor4all-rs/`
3. clone from GitHub at the pinned fallback in [deps/build.jl](deps/build.jl)

Run it with the package project:

```bash
julia --startup-file=no --project=. deps/build.jl
```

## Tests

```julia
using Pkg
Pkg.activate("/path/to/Tensor4all.jl")
Pkg.test()
```

Direct `julia test/runtests.jl` runs also work. HDF5 tests can be skipped
explicitly with `T4A_SKIP_HDF5_TESTS=1`.
