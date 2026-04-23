# Tensor4all.jl

Julia frontend for [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs).

## Current Direction

This repository now follows the implementation-phase Julia-facing architecture:

- `TensorNetworks.TensorTrain` is the indexed chain type
- `SimpleTT.TensorTrain{T,N}` is the raw-array tensor-train layer
- `TensorCI` returns `SimpleTT`
- `QuanticsTransform` is a Julia-owned operator layer
- `ITensorCompat` is an opt-in migration facade over `TensorNetworks`
- HDF5 roundtrip is handled in pure Julia through `save_as_mps` / `load_tt`

`TreeTensorNetwork` still exists in the repository, but it is no longer the
primary public story.

## Implemented

- pure Julia `Index` and `Tensor` metadata types
- `TensorNetworks.TensorTrain` with `data`, `llim`, and `rlim`
- pure Julia `TensorNetworks` helper surface for site queries, site replacement,
  site regrouping, diagonal helpers, and sparse-site matching
- explicit `Vector{Index}` operator-space setters on
  `TensorNetworks.LinearOperator`
- `SimpleTT` compression with `:LU`, `:CI`, and `:SVD`
- pure Julia MPO-MPO contraction for `SimpleTT` with `:naive` and `:zipup`
- `TensorCI.crossinterpolate2` returning `SimpleTT.TensorTrain`
- `ITensorCompat.MPS` / `MPO` wrappers for migration-oriented chain workflows,
  including raw MPS blocks in `(left, site, right)` order and raw MPO blocks in
  `(left, input, output, right)` order
- pure Julia HDF5 MPS-schema roundtrip through the HDF5 extension
- adopted quantics grid re-exports from `QuanticsGrids.jl`

## Still Missing

- deeper `QuanticsTransform` kernels and validation coverage
- broader `TreeTensorNetwork` / non-chain functionality
- finalized Julia-facing reduced `tensor4all-rs` ABI documentation

## Installation

### Prerequisites

- Julia 1.9 or later
- A Rust toolchain (cargo) — installed automatically via
  [RustToolChain.jl](https://github.com/nicholaskl97/RustToolChain.jl) during
  the build step
- Git (for the GitHub clone fallback if no local `tensor4all-rs` source is
  available)

### Setup

```bash
# Clone the repository
git clone https://github.com/tensor4all/Tensor4all.jl.git
cd Tensor4all.jl

# Install Julia dependencies
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'

# Build the Rust backend library
julia --startup-file=no --project=. deps/build.jl

# Verify the installation
julia --startup-file=no --project=. -e 'using Tensor4all; println("Tensor4all loaded successfully")'
```

If you have a local `tensor4all-rs` checkout, point to it before building:

```bash
export TENSOR4ALL_RS_PATH=/path/to/tensor4all-rs
julia --startup-file=no --project=. deps/build.jl
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution flow (issue →
spec → plan → implementation).

## Development Notes

- `tensor4all-rs` remains the backend for low-level kernels.
- `Tensor4all.jl` owns the pure Julia public object model.
- `TensorCrossInterpolation.jl` is used as an implementation dependency for the
  current `SimpleTT` / `TensorCI` implementation.

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

Direct `julia test/runtests.jl` runs also work. HDF5 extension tests are
skipped automatically in direct runs when `HDF5` is not visible in the active
project, and can also be skipped explicitly with `T4A_SKIP_HDF5_TESTS=1`.
