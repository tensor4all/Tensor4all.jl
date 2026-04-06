# Tensor4all.jl

Julia wrapper for the [tensor4all](https://github.com/tensor4all/tensor4all-rs) Rust library.

Provides tensor network types compatible with [ITensors.jl](https://github.com/ITensor/ITensors.jl), backed by efficient Rust implementations.

## Prerequisites

### Rust

If Rust is not installed, run:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, open a new terminal or run `source ~/.cargo/env`.

## Installation

The Rust shared library is compiled automatically by `Pkg.build()`.

### Option 1: Develop locally

```julia
using Pkg
Pkg.activate("/path/to/Tensor4all.jl")
Pkg.build()
```

### Option 2: Develop from another environment

```julia
using Pkg
Pkg.develop(path="/path/to/Tensor4all.jl")
Pkg.build("Tensor4all")
```

### Option 3: Add from GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl.git")
Pkg.build("Tensor4all")
```

The built shared library lives in `deps/libtensor4all_capi.{dylib,so,dll}` inside
the package directory.

### Rust source resolution

`deps/build.jl` looks for `tensor4all-rs` in this order:

1. `TENSOR4ALL_RS_PATH`
2. sibling directory `../tensor4all-rs/`
3. clone from GitHub

If you run the build script directly, use the package project:

```bash
julia --startup-file=no --project=. deps/build.jl
```

## Running Tests

```julia
using Pkg
Pkg.activate("/path/to/Tensor4all.jl")
Pkg.test()
```

To skip HDF5 tests, set `T4A_SKIP_HDF5_TESTS=1`.

## Modules

- `Tensor4all`: `Index`, `Tensor`, `onehot`, HDF5 save/load
- `Tensor4all.SimpleTT`: simple tensor trains with fixed site dimensions
- `Tensor4all.TreeTN`: MPS/MPO/tree tensor network operations
- `Tensor4all.QuanticsGrids`: coordinate transforms between physical and quantics grids
- `Tensor4all.QuanticsTransform`: quantics shift/flip/phase/cumsum/Fourier/affine operators
- `Tensor4all.QuanticsTCI`: quantics tensor cross interpolation
- `Tensor4all.TreeTCI`: tree-structured tensor cross interpolation

## Usage

```julia
using Tensor4all
```

### Index and Tensor

```julia
i = Index(2)
j = Index(3; tags="Site,n=1")

dim(i)            # 2
tags(j)           # "Site,n=1"
hastag(j, "Site") # true

t = Tensor([i, j], rand(2, 3))
rank(t)           # 2
dims(t)           # (2, 3)
storage_kind(t)   # DenseF64
indices(t)        # [i, j]

z = Tensor([i, j], rand(ComplexF64, 2, 3))
oh = onehot(i => 1, j => 2)
```

### Simple Tensor Trains

```julia
using Tensor4all.SimpleTT

tt = SimpleTensorTrain([2, 3, 4], 1.0)

tt(0, 0, 0)                          # 1.0
Tensor4all.SimpleTT.site_dims(tt)    # [2, 3, 4]
Tensor4all.SimpleTT.link_dims(tt)    # []
Tensor4all.SimpleTT.site_tensor(tt, 0)
sum(tt)
```

### Tree Tensor Networks

```julia
using Tensor4all.TreeTN

sites = [Index(2) for _ in 1:5]
mps = random_mps(sites; linkdims=4)

length(mps)        # 5
nv(mps)            # 5
ne(mps)            # 4
maxbonddim(mps)    # 4

orthogonalize!(mps, 3)
truncate!(mps; maxdim=2)
inner(mps, mps)
to_dense(mps)
```

`SimpleTensorTrain` can also be converted to an MPS:

```julia
using Tensor4all.SimpleTT: SimpleTensorTrain
using Tensor4all.TreeTN: MPS

mps = MPS(SimpleTensorTrain([2, 2, 2], 1.0))
```

### Quantics Grids

```julia
using Tensor4all: DiscretizedGrid, localdimensions
using Tensor4all.QuanticsGrids: origcoord_to_quantics, quantics_to_origcoord

grid = DiscretizedGrid(2, [2, 2], [0.0, 0.0], [1.0, 1.0]; unfolding=:grouped)

q = origcoord_to_quantics(grid, [0.25, 0.75])
x = quantics_to_origcoord(grid, q)

length(q)              # 4
x                       # approximately [0.25, 0.75]
localdimensions(grid)   # [2, 2, 2, 2]
```

### Quantics Transform

```julia
using Tensor4all.SimpleTT: SimpleTensorTrain
using Tensor4all.TreeTN: MPS
using Tensor4all.QuanticsTransform

mps = MPS(SimpleTensorTrain([2, 2, 2], 1.0))

op = shift_operator(3, 1)
set_iospaces!(op, mps)
shifted = apply(op, mps; method=:naive)

multi = shift_operator_multivar(3, 1, 2, 0)
flipped = flip_operator_multivar(3, 2, 1; bc=Open)
phase = phase_rotation_operator_multivar(3, pi / 4, 2, 1)
aff = affine_operator(
    3,
    Int64[1 -1; 1 0; 0 1],
    ones(Int64, 3, 2),
    Int64[0, 0, 0],
    ones(Int64, 3);
    bc=[Open, Periodic, Periodic],
)
```

### Interpolation Modules

High-level interpolation APIs are available in:

- `Tensor4all.QuanticsTCI`
- `Tensor4all.TreeTCI`

See the module docstrings in `src/QuanticsTCI.jl` and `src/TreeTCI.jl` for the
current entry points.

### HDF5 Save/Load

Tensors and MPS can be saved to HDF5 files in a format compatible with ITensors.jl.

```julia
# Save/load a single tensor
save_itensor("data.h5", "my_tensor", t)
t_loaded = load_itensor("data.h5", "my_tensor")

# Save/load an MPS
using Tensor4all.TreeTN: save_mps, load_mps

save_mps("data.h5", "my_mps", mps)
mps_loaded = load_mps("data.h5", "my_mps")
```

### ITensors.jl Integration

When ITensors.jl is loaded, bidirectional conversion is available via the package extension:

```julia
using Tensor4all
using ITensors

# Tensor4all.Index → ITensors.Index
t4a_idx = Tensor4all.Index(4; tags="Site")
it_idx  = convert(ITensors.Index, t4a_idx)

# ITensors.Index → Tensor4all.Index
it_idx2  = ITensors.Index(3, "Link")
t4a_idx2 = convert(Tensor4all.Index, it_idx2)

# Same conversions work for Tensor ↔ ITensor
```

## Debugging

### Rust backtraces

When a Rust-side error occurs, Tensor4all.jl automatically includes a Rust
backtrace in the error message (via `RUST_BACKTRACE=1`).  The release build
ships with debug info (`[optimized + debuginfo]`), so backtraces show file
names and line numbers:

```
ERROR: Tensor4all C API error: Internal error: Invalid pivot: ...

Rust backtrace:
   0: tensor4all_capi::set_last_error
             at .../src/lib.rs:91:14
   1: tensor4all_capi::err_status
             at .../src/lib.rs:105:5
   2: tensor4all_capi::tensorci::t4a_crossinterpolate2_f64::{{closure}}
             at .../src/tensorci.rs:477:23
   ...
```

You can control this behaviour with the `RUST_BACKTRACE` environment variable:

| Value   | Effect |
|---------|--------|
| *(unset)* | Tensor4all.jl sets it to `1` automatically |
| `0`     | Disable Rust backtraces |
| `1`     | Show backtraces with file/line info (default) |
| `full`  | Show backtraces with additional detail |

### Debug build

By default, `Pkg.build` compiles the Rust backend in release mode (optimised +
debug info).  To build without optimisations (for full backtrace fidelity), set:

```bash
TENSOR4ALL_BUILD_DEBUG=1 julia -e 'using Pkg; Pkg.build("Tensor4all")'
```

## Troubleshooting

### Error: "tensor4all-capi library not found"

The shared library has not been built. Run:

```julia
Pkg.build("Tensor4all")
```

Then verify the library exists:
- macOS: `deps/libtensor4all_capi.dylib`
- Linux: `deps/libtensor4all_capi.so`

### Error: "Could not find cargo"

Rust is not installed or not in PATH. Install Rust (see Prerequisites above), then open a new terminal.
