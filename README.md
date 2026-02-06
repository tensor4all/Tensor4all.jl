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

The Rust shared library is **automatically compiled** by `Pkg.build()`. No manual build steps are required.

### Option 1: Develop locally (for package development)

```julia
using Pkg
Pkg.activate("/path/to/Tensor4all.jl")
Pkg.build()  # Automatically compiles the Rust backend
```

The shared library is installed to:
```
/path/to/Tensor4all.jl/deps/libtensor4all_capi.{dylib,so,dll}
```

### Option 2: Develop from another environment or global

```julia
using Pkg
Pkg.develop(path="/path/to/Tensor4all.jl")
Pkg.build("Tensor4all")  # Automatically compiles the Rust backend
```

The shared library is installed to:
```
/path/to/Tensor4all.jl/deps/libtensor4all_capi.{dylib,so,dll}
```
(Same location as the source — `Pkg.develop` symlinks to the local directory.)

### Option 3: Add from another environment (e.g., from GitHub URL)

```julia
using Pkg
Pkg.add(url="https://github.com/tensor4all/Tensor4all.jl.git")
Pkg.build("Tensor4all")  # Automatically compiles the Rust backend
```

The shared library is installed to:
```
~/.julia/packages/Tensor4all/<hash>/deps/libtensor4all_capi.{dylib,so,dll}
```

### Rust source resolution

The build script locates the `tensor4all-rs` Rust workspace in this priority order:

1. `TENSOR4ALL_RS_PATH` environment variable
2. Sibling directory `../tensor4all-rs/` (relative to the package root)
3. Automatic clone from GitHub

## Running Tests

```julia
using Pkg
Pkg.activate("/path/to/Tensor4all.jl")
Pkg.test()
```

To skip HDF5 tests: set `T4A_SKIP_HDF5_TESTS=1` before running.

## Usage

```julia
using Tensor4all
```

### Index

```julia
# Create an index with dimension 5
i = Index(5)

# Create an index with tags
j = Index(3; tags="Site,n=1")

# Access properties
dim(i)            # 5
id(i)             # unique UInt64 ID
tags(j)           # "Site,n=1"
hastag(j, "Site") # true

# Copy an index (same ID)
j2 = copy(j)

# Create a similar index (new ID, same dim and tags)
j3 = sim(j)
```

### Tensor

```julia
i = Index(2)
j = Index(3)

# Create a dense Float64 tensor
data = rand(2, 3)
t = Tensor([i, j], data)

# Access properties
rank(t)          # 2
dims(t)          # (2, 3)
storage_kind(t)  # DenseF64
indices(t)       # [i, j]

# Retrieve data (column-major Julia array)
retrieved = Tensor4all.data(t)

# Retrieve data in a specific index order
arr = Array(t, [j, i])  # shape (3, 2), transposed

# Create a complex tensor
z = Tensor([i, j], rand(ComplexF64, 2, 3))

# Create a higher-rank tensor
k = Index(4)
t3 = Tensor([i, j, k], rand(2, 3, 4))

# Create a one-hot tensor
oh = onehot(i => 1, j => 2)  # 1.0 at position [1, 2]
```

### MPS (Matrix Product State) / Tensor Train

```julia
using Tensor4all.TreeTN

# Create a random MPS
sites = [Index(2) for _ in 1:5]
mps = random_mps(sites; linkdims=4)

# Properties
length(mps)       # 5
nv(mps)           # 5 (number of vertices)
ne(mps)           # 4 (number of edges)
maxbonddim(mps)   # 4
linkdims(mps)     # [4, 4, 4, 4]

# Access tensors (1-indexed)
mps[1]            # first tensor
collect(mps)      # all tensors as a vector

# Orthogonalize (QR-based canonical form)
orthogonalize!(mps, 3)
canonical_form(mps)  # Unitary

# Other canonical forms: LU, CI
orthogonalize!(mps, 1; form=LU)

# Truncate bond dimensions
truncate!(mps; maxdim=2)

# Inner product and norm
ip = inner(mps, mps)
n  = norm(mps)

# Contract to a dense tensor
dense = to_dense(mps)

# Contract two MPS/MPO
result = contract(mps_a, mps_b)                     # zipup (default)
result = contract(mps_a, mps_b; method=:fit)         # fit
result = contract(mps_a, mps_b; method=:naive)       # naive
```

### Tensor Cross Interpolation (TCI)

```julia
using Tensor4all.TensorCI
using Tensor4all.SimpleTT

# Approximate a function as a tensor train
f(i, j, k) = Float64((1 + i) * (1 + j) * (1 + k))  # 0-based indices
tt, err = crossinterpolate2(f, [3, 4, 5]; tolerance=1e-10)

# Evaluate the tensor train
tt(0, 0, 0)  # ≈ 1.0
tt(2, 3, 4)  # ≈ 60.0

# Sum over all elements
sum(tt)
```

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
