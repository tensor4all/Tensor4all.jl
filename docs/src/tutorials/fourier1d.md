# 1D Fourier Transform

This tutorial demonstrates how to compute the **discrete Fourier transform
(DFT)** of a function on an exponentially large quantics grid using
Tensor4all.jl.  The key idea is that the DFT matrix itself admits a
low-rank tensor train (QTT) representation, so the transform can be carried
out entirely within the QTT framework -- without ever constructing the full
``2^R \times 2^R`` DFT matrix.

## Why QTT-based Fourier transforms?

A standard DFT on ``M = 2^R`` points costs ``O(M \log M)`` via FFT.  When
``R = 40``, we have ``M \approx 10^{12}`` -- a trillion points -- and even
the FFT becomes infeasible.  The QTT approach bypasses this: if the input
function has a compact QTT representation (small bond dimension), and the DFT
operator also has small bond dimension (it does), then the Fourier
coefficients can be computed by a tensor-train contraction whose cost scales
only with the bond dimensions, not with ``M``.

This is particularly valuable when you need Fourier coefficients at
*specific* frequencies (small ``k`` or large ``k``) rather than the entire
spectrum, since individual coefficients can be evaluated in ``O(R)`` time
once the output QTT is available.

## Setup

```julia
using Tensor4all.QuanticsGrids       # Grid types and coordinate conversions
using Tensor4all.QuanticsTCI         # quanticscrossinterpolate
using Tensor4all.SimpleTT            # to_tensor_train
using Tensor4all.TreeTN              # MPS, linkdims, ...
using Tensor4all.QuanticsTransform   # fourier_operator, set_iospaces!, apply
using Tensor4all.TreeTCI: evaluate as ttn_evaluate  # evaluate TTN at quantics index
```

## Problem: Fourier transform of a sum of exponentials

We consider a function on ``[0, 1]`` that mimics a bosonic correlator:

```math
f(x) = \sum_p \frac{c_p}{1 - e^{-\varepsilon_p}}\, e^{-\varepsilon_p\, x}
```

with parameters ``c_1 = 1.5``, ``\varepsilon_1 = 3.0``, ``c_2 = 0.7``,
``\varepsilon_2 = 15.0``.  This function varies rapidly near ``x = 0`` (due
to the ``\varepsilon_2 = 15`` term) and has a slow exponential tail,
creating multi-scale structure ideally suited for quantics representation.

```julia
c  = [1.5, 0.7]
eps = [3.0, 15.0]

f(x) = Base.sum(c[p] * exp(-eps[p] * x) / (1 - exp(-eps[p])) for p in 1:2)
```

!!! note
    We write `Base.sum` instead of `sum` because `QuanticsTCI` also exports
    a `sum` function.  Using the qualified name avoids ambiguity.

## Step 1: Quantics cross-interpolation

Create a quantics grid with ``R = 40`` bits (``2^{40} \approx 10^{12}``
points) and build the QTT approximation:

```julia
R = 40
grid = DiscretizedGrid(1, R, [0.0], [1.0])

ci, ranks, errors = quanticscrossinterpolate(grid, f; tolerance=1e-10)
```

The maximum bond dimension will be small (typically around 10--15),
reflecting the low complexity of exponentials in the quantics representation.

## Step 2: Convert to MPS

Convert the QTCI result to a `TreeTensorNetwork` (MPS) for use with the
Fourier operator:

```julia
tt = to_tensor_train(ci)   # SimpleTensorTrain
fmps = MPS(tt)             # TreeTensorNetwork{Int}
```

## Step 3: Construct and apply the Fourier operator

The `fourier_operator` function creates the QTT representation of the DFT
matrix with the convention

```math
\hat{f}_k = \frac{1}{\sqrt{M}} \sum_{m=0}^{M-1} f_m\, e^{-2\pi i\, k m / M},
\qquad M = 2^R.
```

Before applying the operator, its site indices must be bound to those of the
input MPS via `set_iospaces!`:

```julia
ft_op = fourier_operator(R; forward=true)
set_iospaces!(ft_op, fmps)
hfmps = apply(ft_op, fmps; method=:naive)
```

!!! tip
    The `method=:naive` contraction is the simplest and works well when
    bond dimensions are moderate.  For larger problems, `:zipup` or `:fit`
    can reduce the bond dimension of the result at the cost of a controlled
    approximation error.

## Step 4: Evaluate Fourier coefficients

To evaluate ``\hat{f}_k`` at a specific frequency index ``k``, convert
``k`` to its binary (quantics) representation and call `ttn_evaluate`:

```julia
k = 5
bits = digits(k, base=2, pad=R)   # LSB-first bit vector, length R
val = ttn_evaluate(hfmps, bits)
```

The `digits` function returns the bits in LSB-first order, which is the
correct convention for `ttn_evaluate` -- no reversal is needed.

!!! note
    The result `hfmps` from `apply` is a `TreeTensorNetwork` that may not
    satisfy `is_chain(hfmps)`.  Always use `ttn_evaluate` to read out
    values, rather than assuming a chain topology.

## Step 5: Compare with the analytical DFT

Each exponential term ``g_p(x) = c_p / (1 - e^{-\varepsilon_p})\, e^{-\varepsilon_p\, x}``
sampled at ``x = m/M`` gives a geometric series in the DFT.  The exact
discrete Fourier coefficient is:

```math
\hat{f}_k = \frac{1}{\sqrt{M}} \sum_p
  \frac{c_p}{1 - e^{-\varepsilon_p}}
  \cdot \frac{1 - e^{-(\varepsilon_p + 2\pi i k)}}{1 - e^{-(\varepsilon_p + 2\pi i k)/M}}
```

This follows from summing the geometric series
``\sum_{m=0}^{M-1} r^m = (1 - r^M)/(1 - r)`` with
``r = e^{-(\varepsilon_p + 2\pi i k)/M}``.

In Julia:

```julia
M = 2^R

function analytical_dft(k)
    result = zero(ComplexF64)
    for p in 1:2
        z = eps[p] + 2π * im * k
        geom = (1 - exp(-z)) / (1 - exp(-z / M))
        result += c[p] / (1 - exp(-eps[p])) * geom
    end
    return result / sqrt(M)
end
```

Check the accuracy at a few frequencies:

```julia
for k in [0, 1, 5, 100]
    bits = digits(k, base=2, pad=R)
    qft_val = ttn_evaluate(hfmps, bits)
    exact_val = analytical_dft(k)
    rel_err = abs(qft_val - exact_val) / abs(exact_val)
    println("k = $k:  QFT = $qft_val,  exact = $exact_val,  rel_err = $rel_err")
end
```

## High-frequency accuracy

A major advantage of the QTT Fourier transform is access to high-frequency
coefficients at no additional cost.  Evaluating ``\hat{f}_{10000}`` is just
as cheap as evaluating ``\hat{f}_1``:

```julia
for k in [10, 100, 1000, 10000]
    bits = digits(k, base=2, pad=R)
    qft_val = ttn_evaluate(hfmps, bits)
    exact_val = analytical_dft(k)
    rel_err = abs(qft_val - exact_val) / abs(exact_val)
    println("k = $k:  rel_err = $rel_err")
end
```

With tolerance ``10^{-10}`` in the QTCI step, relative errors are typically
``10^{-10}`` or better across all frequencies.

## Complete example

Putting it all together:

```julia
using Tensor4all.QuanticsGrids
using Tensor4all.QuanticsTCI
using Tensor4all.SimpleTT
using Tensor4all.TreeTN
using Tensor4all.QuanticsTransform
using Tensor4all.TreeTCI: evaluate as ttn_evaluate

# Parameters
c   = [1.5, 0.7]
eps = [3.0, 15.0]
R   = 40
M   = 2^R

# Target function
f(x) = Base.sum(c[p] * exp(-eps[p] * x) / (1 - exp(-eps[p])) for p in 1:2)

# 1. QTCI on a trillion-point grid
grid = DiscretizedGrid(1, R, [0.0], [1.0])
ci, ranks, errors = quanticscrossinterpolate(grid, f; tolerance=1e-10)

# 2. Convert to MPS
tt = to_tensor_train(ci)
fmps = MPS(tt)

# 3. Apply Fourier operator
ft_op = fourier_operator(R; forward=true)
set_iospaces!(ft_op, fmps)
hfmps = apply(ft_op, fmps; method=:naive)

# 4. Analytical reference (exact geometric series)
function analytical_dft(k)
    result = zero(ComplexF64)
    for p in 1:2
        z = eps[p] + 2π * im * k
        geom = (1 - exp(-z)) / (1 - exp(-z / M))
        result += c[p] / (1 - exp(-eps[p])) * geom
    end
    return result / sqrt(M)
end

# 5. Compare at low and high k
for k in [0, 1, 5, 100, 10000]
    bits = digits(k, base=2, pad=R)
    qft_val = ttn_evaluate(hfmps, bits)
    exact_val = analytical_dft(k)
    rel_err = abs(qft_val - exact_val) / abs(exact_val)
    println("k = $k:  rel_err = $(round(rel_err, sigdigits=2))")
end
```

## The pipeline at a glance

The overall workflow is:

```
f(x)  →  QTCI  →  MPS  →  Fourier operator × MPS  →  evaluate at k
```

Each step is efficient: the QTCI scales with bond dimension, not grid size;
the Fourier operator has ``O(1)`` bond dimension per site; the contraction
is a standard tensor-train operation; and pointwise evaluation costs
``O(R)``.

## API summary

| Function | Description |
|----------|-------------|
| `quanticscrossinterpolate(grid, f; tolerance)` | Build QTT via cross-interpolation |
| `to_tensor_train(ci)` | Convert QTCI result to `SimpleTensorTrain` |
| `MPS(tt)` | Convert `SimpleTensorTrain` to `TreeTensorNetwork{Int}` |
| `fourier_operator(R; forward=true)` | Create QTT Fourier operator (`LinearOperator`) |
| `set_iospaces!(op, mps)` | Bind operator site indices to the input MPS |
| `apply(op, mps; method=:naive)` | Apply operator to MPS, return `TreeTensorNetwork` |
| `ttn_evaluate(ttn, bits)` | Evaluate TTN at a quantics bit vector |
| `digits(k, base=2, pad=R)` | Convert integer `k` to LSB-first bit vector |

### Normalization convention

The forward Fourier operator implements:

```math
\hat{f}_k = \frac{1}{\sqrt{M}} \sum_{m=0}^{M-1} f_m\, e^{-2\pi i\, k m / M}
```

This is the **symmetric** convention with ``1/\sqrt{M}`` normalization and a
**negative** sign in the exponent.  When comparing with analytical results,
ensure your reference formula uses the same convention.
