# Tensor4all.jl

Julia frontend for [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs).

## Current Direction

This branch restores the older Julia-facing architecture and treats it as an
implementation target:

- `TensorNetworks.TensorTrain` is the indexed chain type
- `SimpleTT.TensorTrain{T,N}` is the raw-array tensor-train layer
- `TensorCI.crossinterpolate2` returns `TensorCI2`
- `SimpleTT.TensorTrain(tci)` is the conversion boundary into the raw-array TT layer
- `QuanticsTransform` is a Julia-owned operator layer
- HDF5 roundtrip is handled directly through `HDF5.jl`

`TreeTensorNetwork` still exists in the repository, but it is not the primary
public story of this branch.

## Implemented Today

- backend-backed `Index` and `Tensor` wrappers with Julia-side metadata accessors
- `TensorNetworks.TensorTrain` and `TensorNetworks.LinearOperator` containers
- explicit `set_input_space!`, `set_output_space!`, and `set_iospaces!` on `Vector{Index}`
- `SimpleTT` compression with `:LU`, `:CI`, and `:SVD`
- pure Julia MPO-MPO contraction for `SimpleTT` with `:naive` and `:zipup`
- `TensorCI.crossinterpolate2` returning `TensorCI2`
- `SimpleTT.TensorTrain(tci)` conversion into the raw-array TT layer
- HDF5.jl-backed MPS-schema roundtrip through `save_as_mps` / `load_tt`
- adopted quantics grid re-exports from `QuanticsGrids.jl`

## Still Missing

- full docs cleanup beyond the restored architecture story
- finalized Julia-facing reduced `tensor4all-rs` ABI documentation
- executable `TensorNetworks.apply` and the remaining site helper utilities
- Rust-aligned `QuanticsTransform` validation and operator kernels
- broader `TreeTensorNetwork` / non-chain functionality

## Development Notes

- `tensor4all-rs` remains the backend for low-level kernels.
- `Tensor4all.jl` owns the pure Julia public object model.
- `TensorCrossInterpolation.jl` is used as an implementation dependency for the
  current `SimpleTT` / `TensorCI` boundary.

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
