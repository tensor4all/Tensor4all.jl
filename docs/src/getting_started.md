# Getting Started

This page walks through the basic building blocks of Tensor4all.jl.

## Loading the Package

```julia
using Tensor4all
```

All core types (`Index`, `Tensor`) and their accessors (`dim`, `tags`, `plev`,
`inds`, `rank`, `dims`, etc.) are exported at the top level. Submodules are
accessed through qualified names, e.g. `Tensor4all.TensorNetworks`,
`Tensor4all.SimpleTT`, `Tensor4all.TensorCI`.

## Index and Tensor

An `Index` represents a single tensor leg with a dimension, string tags, and a
prime level:

```julia
i = Index(4; tags=["x"], plev=0)
j = Index(3; tags=["y"])

dim(i)    # 4
tags(i)   # ["x"]
plev(j)   # 0

# Prime levels
i' = prime(i)      # plev = 1
i0 = noprime(i')   # plev = 0
```

A `Tensor` wraps a dense Julia `Array` together with `Index` metadata:

```julia
i = Index(2; tags=["i"])
j = Index(3; tags=["j"])

data = reshape(collect(1.0:6.0), 2, 3)
t = Tensor(data, [i, j])

rank(t)   # 2
dims(t)   # (2, 3)
inds(t)   # [Index(2|i; plev=0), Index(3|j; plev=0)]
```

The array dimensions must match the index dimensions exactly, otherwise a
`DimensionMismatch` error is raised.

### Tensor contraction

`contract(::Tensor, ::Tensor)` contracts two tensors over their shared
indices via the backend `t4a_tensor_contract` kernel:

```julia
i = Index(2; tags=["i"])
j = Index(3; tags=["j"])
k = Index(4; tags=["k"])

a = Tensor(randn(2, 3), [i, j])
b = Tensor(randn(3, 4), [j, k])
c = contract(a, b)        # contracts over j; result has [i, k]
inds(c)                   # [Index(2|i; plev=0), Index(4|k; plev=0)]
```

### Tensor SVD / QR

```julia
U, S, V = svd(t, [i]; threshold=1e-10)   # left=[i], right=remaining
Q, R   = qr(t, [i])                       # same left partitioning
```

`svd` takes the same `threshold` / `maxdim` / `svd_policy` truncation
contract as the chain-level operations; see the
[Truncation Policy](truncation_policy.md) page.

## Two TensorTrain Types

Tensor4all.jl provides two separate tensor-train types for different purposes:

| Type | Purpose | Data storage |
|------|---------|-------------|
| `TensorNetworks.TensorTrain` | Indexed chain with metadata (site queries, HDF5 I/O) | `Vector{Tensor}` |
| `SimpleTT.TensorTrain{T,N}` | Raw numerical operations (compression, MPO contraction, TCI output) | `Vector{Array{T,N}}` |

These are independent types with no automatic conversion between them. Use
`TensorNetworks.TensorTrain` when you need index-aware operations and
`SimpleTT.TensorTrain` when you need numerical algorithms.

## Building an Indexed TensorTrain

`TensorNetworks.TensorTrain` stores a chain of `Tensor` objects. Each tensor
has site indices (physical legs) and link indices (bond legs connecting
neighboring tensors).

### Conventions

- **Link indices** must be tagged with `"Link"` (e.g.
  `Index(4; tags=["Link", "l=1"])`).
- **Site indices** use any other tags (e.g. `Index(2; tags=["x", "x=1"])`).
- **`llim`** and **`rlim`** track the orthogonality center boundaries.
  The convenience constructor `TensorTrain(data)` sets `llim=0` and
  `rlim=length(data)+1` (no orthogonality guaranteed).

### MPS-like example (one site index per tensor)

```julia
const TN = Tensor4all.TensorNetworks

# Site indices (physical legs)
s1 = Index(2; tags=["x", "x=1"])
s2 = Index(2; tags=["x", "x=2"])
s3 = Index(2; tags=["x", "x=3"])

# Link indices (bond legs) — must be tagged "Link"
l1 = Index(3; tags=["Link", "l=1"])
l2 = Index(3; tags=["Link", "l=2"])

# Build tensors: first site has shape (site, bond),
# middle sites have shape (bond, site, bond),
# last site has shape (bond, site).
t1 = Tensor(randn(2, 3), [s1, l1])
t2 = Tensor(randn(3, 2, 3), [l1, s2, l2])
t3 = Tensor(randn(3, 2), [l2, s3])

tt = TN.TensorTrain([t1, t2, t3])
length(tt)  # 3
```

### Querying sites

```julia
TN.findsite(tt, s2)                        # 2
TN.findsites(tt, [s1, s3])                 # [1, 3]
TN.findallsiteinds_by_tag(tt; tag="x")    # [s1, s2, s3] (ordered by tag number)
TN.findallsites_by_tag(tt; tag="x")       # [1, 2, 3]
```

## Chain operations on `TensorNetworks.TensorTrain`

The following all accept the common `(threshold, maxdim, svd_policy)`
truncation kwargs. See the [Truncation Policy](truncation_policy.md) page
for the decision rules, the default registry
(`set_default_svd_policy!` / `with_svd_policy`), and the ITensors.jl-
compatible preset.

### Truncation

```julia
# Drop singular values with σ / σ_max ≤ 1e-8 (default policy)
tt_trunc = TN.truncate(tt; threshold=1e-8)

# Cap the bond dimension regardless of threshold
tt_trunc = TN.truncate(tt; maxdim=32)

# Combine both knobs
tt_trunc = TN.truncate(tt; threshold=1e-8, maxdim=32)
```

### Addition

```julia
# Elementwise sum of two TensorTrains with matching site indices
tt_sum = TN.add(tt_a, tt_b; threshold=1e-10)
```

### Contraction

```julia
# Contract two TensorTrains over their shared site indices
tt_c = TN.contract(tt_a, tt_b; threshold=1e-8, maxdim=64)

# Method choices: :zipup (default), :fit, :naive
tt_c = TN.contract(tt_a, tt_b; method=:fit, nfullsweeps=4)
```

### Applying a LinearOperator

`TN.LinearOperator` wraps an MPO-like `TensorTrain` together with explicit
input/output `Index` bindings; `TN.apply` evaluates it against a chain state.
`QuanticsTransform` ships constructors (`shift_operator`, `affine_operator`,
`binaryop_operator`, etc.) that return `LinearOperator` values.

```julia
using Tensor4all.QuanticsTransform
op = shift_operator(4, 1)                    # 4-bit quantics shift by +1
sites = op.input_indices
state = TN.random_tt(sites; linkdims=4)
TN.set_iospaces!(op, op.input_indices, op.output_indices)
result = TN.apply(op, state; threshold=1e-10)
```

### Linear solve

```julia
# Solve A x = b where A is a LinearOperator and b is a TensorTrain
x = TN.linsolve(A, b; threshold=1e-10, maxdim=64, nfullsweeps=8)
```

### Reorganizing site-index topology

```julia
# Fuse adjacent nodes into coarser groups
tt_fused = TN.fuse_to(tt, [[s1, s2], [s3]])

# Split a node into multiple target nodes (with optional final truncation)
tt_split = TN.split_to(tt, [[s1], [s2, s3]]; threshold=1e-8, final_sweep=true)

# General restructuring (dispatches to fuse / split / swap)
tt_r = TN.restructure_to(tt, target_groups;
    split_threshold=1e-8, final_threshold=1e-8)
```

### Norms, inner products, comparison

```julia
n = TN.norm(tt)                     # Frobenius norm
d = TN.dot(tt_a, tt_b)              # inner product
d = TN.dist(tt_a, tt_b)             # Euclidean distance
is_eq = isapprox(tt_a, tt_b; rtol=1e-10)
```

## SimpleTT: Compression and Contraction

`SimpleTT.TensorTrain{T,N}` works with raw Julia arrays. The array layout
convention is:

- **`N=3` (MPS-like):** each site tensor has shape
  `(bond_left, site_dim, bond_right)`. The first site has `bond_left = 1` and
  the last site has `bond_right = 1`.
- **`N=4` (MPO-like):** each site tensor has shape
  `(bond_left, site_in, site_out, bond_right)`.

### Compression

```julia
# Build a simple 2-site MPS with bond dimension 2
tt = Tensor4all.SimpleTT.TensorTrain([
    reshape([1.0, 0.0, 0.0, 1.0], 1, 2, 2),   # shape (1, 2, 2)
    reshape([1.0, 0.0, 0.0, 1e-15], 2, 2, 1),  # shape (2, 2, 1)
])

# Compress using SVD truncation (relative tolerance on singular values)
Tensor4all.SimpleTT.compress!(tt, :SVD; tolerance=1e-12)

size(tt.sitetensors[1])  # (1, 2, 1) — bond dimension reduced from 2 to 1
size(tt.sitetensors[2])  # (1, 2, 1)
```

Supported compression methods: `:SVD`, `:LU`, `:CI`.

Optional keyword arguments: `tolerance` (relative tolerance, default `1e-12`),
`maxbonddim` (maximum bond dimension, default unlimited).

### MPO-MPO contraction

```julia
# Two MPO-like tensor trains (N=4)
a = Tensor4all.SimpleTT.TensorTrain([
    reshape(Float64[1, 0, 0, 1], 1, 2, 2, 1),
    reshape(Float64[1, 0, 0, 1], 1, 2, 2, 1),
])
b = Tensor4all.SimpleTT.TensorTrain([
    reshape(Float64[1, 0, 0, 1], 1, 2, 2, 1),
    reshape(Float64[1, 0, 0, 1], 1, 2, 2, 1),
])

result = Tensor4all.SimpleTT.contract(a, b; algorithm=:naive)
# or: algorithm=:zipup for zip-up contraction with on-the-fly compression
```

## TensorCI: Cross Interpolation

`TensorCI.crossinterpolate2` approximates a function as a tensor train via
tensor cross interpolation. It returns a `TensorCI2` object that can be
converted to `SimpleTT.TensorTrain`.

```julia
# Approximate f(v1, v2) where v1 ∈ {1,2}, v2 ∈ {1,2}
f(v) = Float64(v[1] == 1 && v[2] == 1)

tci = Tensor4all.TensorCI.crossinterpolate2(
    Float64,    # element type
    f,          # function taking a vector of indices (1-based)
    [2, 2];     # local dimensions for each site
    tolerance=1e-12,
    maxbonddim=10,
)

# Convert to SimpleTT for further manipulation
tt = Tensor4all.SimpleTT.TensorTrain(tci)
length(tt)               # 2
size(tt.sitetensors[1])  # (1, 2, bonddim)
```

!!! note "Minimum 2 sites required"
    `crossinterpolate2` requires at least 2 local dimensions. Passing a single
    dimension raises `ArgumentError`.

The keyword arguments (`tolerance`, `maxbonddim`, etc.) are passed through to
[TensorCrossInterpolation.jl](https://github.com/tensor4all/TensorCrossInterpolation.jl).
See its documentation for the full list of supported options.

## HDF5 Save and Load

The HDF5 extension provides persistence for `TensorNetworks.TensorTrain`.
Activate it by loading `HDF5`:

```julia
using Tensor4all
using HDF5  # activates the HDF5 extension

const TN = Tensor4all.TensorNetworks

# Build a TensorTrain
s1 = Index(2; tags=["Site", "n=1"])
s2 = Index(2; tags=["Site", "n=2"])
t1 = Tensor([1.0, 0.0], [s1])
t2 = Tensor([0.0, 1.0], [s2])
tt = TN.TensorTrain([t1, t2])

# Save
TN.save_as_mps("myfile.h5", "psi", tt)

# Load
tt2 = TN.load_tt("myfile.h5", "psi")
```

The data is written using an MPS-compatible HDF5 schema. `save_as_mps` takes
`(filepath, name, tensor_train)` and `load_tt` takes `(filepath, name)`.
