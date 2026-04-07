# Tensor4all.jl

A Julia wrapper for [tensor4all-rs](https://github.com/tensor4all/tensor4all-rs), providing tensor network operations via C FFI.

## Features

- **Index & Tensor**: Named indices with tags, prime levels, and automatic contraction
- **SimpleTT**: Simple tensor trains with fixed site dimensions
- **TreeTN (MPS/MPO)**: Tree tensor networks — MPS, MPO, and general tree topologies
- **QuanticsGrids**: Coordinate transforms between physical grids and quantics (binary) representation
- **QuanticsTCI**: Tensor cross interpolation in quantics representation
- **QuanticsTransform**: Fourier, shift, flip, phase rotation, and affine operators on MPS
- **TreeTCI**: TCI on arbitrary tree topologies

## Quick Start

```julia
using Tensor4all
using Tensor4all.QuanticsGrids
using Tensor4all.QuanticsTCI
using Tensor4all.TreeTN

# Interpolate a function on a quantics grid
R = 20  # 2^20 grid points
grid = DiscretizedGrid{1}(R, 0.0, 1.0)
f(x) = exp(-10x) * cos(100x)
ci, ranks, errors = quanticscrossinterpolate(Float64, f, grid; tolerance=1e-8)

# Convert to MPS for further operations
tt = to_tensor_train(ci)
mps = MPS(tt)
```

See the tutorials section in the sidebar for cross-module workflows.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl")
```

The Rust backend (`libtensor4all_capi`) is built automatically during `Pkg.build()`.

## Module Overview

See [Module Architecture](@ref) for the dependency graph and data flow.
