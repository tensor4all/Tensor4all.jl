# 1D Quantics Interpolation

This tutorial demonstrates how to use Tensor4all.jl to approximate univariate
functions in the **Quantics Tensor Train (QTT)** representation via
**Tensor Cross Interpolation (TCI)**.

## What is Quantics (QTT) representation?

The quantics representation encodes a function defined on a grid of ``2^R``
points as a tensor train with ``R`` sites, each having local dimension 2
(one qubit).  The key idea is to write the grid index in binary and assign
each bit to a separate tensor site.  Because many physically relevant
functions exhibit multi-scale structure that translates into low-rank
structure in the quantics representation, the resulting tensor train has
small bond dimension -- often logarithmic in the number of grid points.

This makes QTT-based algorithms extremely efficient for functions with
features at widely separated scales: oscillations, sharp peaks, slow
envelopes, and combinations thereof.

## Setup

```julia
using Tensor4all.QuanticsGrids   # Grid types and coordinate conversions
using Tensor4all.QuanticsTCI     # quanticscrossinterpolate, evaluate, integral, ...
using Tensor4all.TreeTN          # MPS, orthogonalize!, truncate!, ...
```

## Warm-up: approximating sin(x)

We start with a simple example to illustrate the basic workflow.

### Create a quantics grid

`DiscretizedGrid` maps a continuous interval onto a uniform grid of ``2^R``
points.  The first argument is the number of dimensions (1 for a univariate
function), the second is the resolution ``R``, and the third/fourth are
vectors giving the lower and upper bounds.

```julia
R = 20   # 2^20 ≈ 1 million grid points
grid = DiscretizedGrid(1, R, [0.0], [2*pi])
```

### Run QTCI

`quanticscrossinterpolate` takes the grid and a callable as its first two
positional arguments.  The function `f` receives one `Float64` argument per
dimension and must return a `Float64`.

```julia
f_sin(x) = sin(x)

ci, ranks, errors = quanticscrossinterpolate(grid, f_sin; tolerance=1e-10)
```

The return values are:

| Value    | Description |
|----------|-------------|
| `ci`     | `QuanticsTensorCI2` object -- the interpolant |
| `ranks`  | `Vector{Int}` -- maximum bond dimension per TCI sweep |
| `errors` | `Vector{Float64}` -- estimated error per sweep |

### Evaluate the interpolant

`evaluate` takes a vector of **1-based grid indices** (one per dimension):

```julia
i = 42
val = evaluate(ci, [i])      # value of the interpolant at grid point i
```

To convert between grid indices and physical coordinates, use the
coordinate conversion functions from `QuanticsGrids`:

```julia
x_vec = grididx_to_origcoord(grid, [i])   # grid index → coordinate vector
x = x_vec[1]                               # extract the scalar

idx_vec = origcoord_to_grididx(grid, [1.0]) # coordinate → grid index vector
```

Putting it together to check the approximation quality:

```julia
for i in [1, 100, 2^R]
    x = grididx_to_origcoord(grid, [i])[1]
    println("x = $x,  QTCI = $(evaluate(ci, [i])),  exact = $(f_sin(x))")
end
```

### Compute the integral

`integral(ci)` returns the integral over the domain (the sum of all grid
values multiplied by the grid spacing):

```julia
I_qtci = integral(ci)
I_exact = -cos(2*pi) + cos(0.0)  # = 0
println("QTCI integral: $I_qtci,  exact: $I_exact")
```

`sum(ci)` returns the plain sum of all ``2^R`` function values (without the
grid-spacing factor):

```julia
S = sum(ci)
```

The two are related by `integral(ci) ≈ sum(ci) * dx` where
`dx = (upper - lower) / 2^R`.

### Inspect bond dimensions

```julia
println("Maximum bond dimension: ", rank(ci))
println("Bond dimensions:        ", link_dims(ci))
```

For `sin(x)` with tolerance `1e-10`, the maximum bond dimension will
typically be quite small (around 5--10), reflecting the low complexity of
the function in the quantics representation.

## Main example: multi-scale oscillatory function

Now consider a function with structure at vastly different scales -- the
kind of problem where QTT really shines:

```math
f(x) = e^{-x}\,\cos\!\Bigl(\frac{x}{B}\Bigr), \qquad B = 2^{-30}.
```

The cosine oscillates on a scale of order ``B \approx 10^{-9}`` while the
envelope ``e^{-x}`` varies on a scale of order 1.  Resolving both scales
on a uniform grid requires ``\sim 2^{40}`` points -- over a trillion --
yet the QTT representation needs only modest bond dimension.

### Define the function and grid

```julia
B = 2^(-30)
f(x) = exp(-x) * cos(x / B)

R = 40
grid = DiscretizedGrid(1, R, [0.0], [1.0])
```

### Run QTCI with a bond-dimension cap

You can limit the maximum bond dimension instead of (or in addition to)
specifying a tolerance:

```julia
ci, ranks, errors = quanticscrossinterpolate(grid, f; max_bonddim=15)
```

Or use a tolerance-based stopping criterion:

```julia
ci2, ranks2, errors2 = quanticscrossinterpolate(grid, f; tolerance=1e-8)
```

!!! tip
    When you are unsure how large the bond dimension needs to be, start with
    a tolerance-based run.  Use `max_bonddim` when you need to control
    computational cost or memory directly.

### Check the approximation

```julia
for i in [1, 2, 3, 2^R]
    x = grididx_to_origcoord(grid, [i])[1]
    println("x = $x,  QTCI = $(evaluate(ci, [i])),  exact = $(f(x))")
end
```

### Coordinate conversions

`origcoord_to_grididx` maps a physical coordinate back to the nearest grid
index.  This is useful for probing the interpolant at a specific point in
the domain:

```julia
x_query = 0.5
i_vec = origcoord_to_grididx(grid, [x_query])
val = evaluate(ci, i_vec)
println("f($x_query) ≈ $val")
```

### Compute the integral

```julia
I_qtci = integral(ci)
println("Integral via QTCI: $I_qtci")
```

### Inspect the TCI convergence

The `ranks` and `errors` vectors track the TCI convergence across sweeps.
They can be used to verify that the interpolation has converged:

```julia
println("Ranks per sweep:  ", ranks)
println("Errors per sweep: ", errors)
println("Final bond dimension: ", rank(ci))
println("Bond dimensions:      ", link_dims(ci))
```

## Converting to MPS

The QTCI result can be converted to an `MPS` (a `TreeTensorNetwork{Int}`)
for further tensor-network operations such as orthogonalization, truncation,
contraction with other MPS/MPO, etc.

The conversion is a two-step process: first extract a `SimpleTensorTrain`,
then wrap it as an `MPS`:

```julia
tt = to_tensor_train(ci)   # SimpleTensorTrain
mps = MPS(tt)              # TreeTensorNetwork{Int}
```

Once you have an MPS, the full TreeTN API is available:

```julia
println("Number of sites: ", nv(mps))
println("Bond dimensions: ", linkdims(mps))

# Orthogonalize toward site 1
orthogonalize!(mps, 1)

# Truncate bond dimensions
truncate!(mps; maxdim=10)
println("Bond dimensions after truncation: ", linkdims(mps))
```

## API summary

| Function | Description |
|----------|-------------|
| `DiscretizedGrid(ndims, R, lo, hi)` | Create a uniform grid with ``2^R`` points per dimension |
| `quanticscrossinterpolate(grid, f; tolerance, max_bonddim)` | Run quantics TCI |
| `evaluate(ci, [i])` | Evaluate the interpolant at 1-based grid index `i` |
| `integral(ci)` | Integral over the continuous domain |
| `sum(ci)` | Sum of all grid-point values |
| `rank(ci)` | Maximum bond dimension |
| `link_dims(ci)` | Vector of bond dimensions |
| `grididx_to_origcoord(grid, [i])` | Grid index to physical coordinate |
| `origcoord_to_grididx(grid, [x])` | Physical coordinate to grid index |
| `to_tensor_train(ci)` | Convert QTCI to `SimpleTensorTrain` |
| `MPS(tt)` | Convert `SimpleTensorTrain` to `TreeTensorNetwork{Int}` |
| `nv(mps)` | Number of vertices |
| `linkdims(mps)` | Bond dimensions of the MPS |
| `orthogonalize!(mps, v)` | Orthogonalize MPS toward vertex `v` |
| `truncate!(mps; maxdim=d)` | Truncate bond dimensions |
