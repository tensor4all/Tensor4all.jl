# Module Overview

## Current Architecture

The restored Julia frontend is layered like this:

```text
Core (Index, Tensor)
  ↓                ↓
TensorNetworks     SimpleTT
  ↓      ↓         ↑
ITensorCompat      TensorCI
  ↓
QuanticsTransform

Adopted wrapper modules:
- QuanticsGrids
- QuanticsTCI
```

`QuanticsGrids.jl` and `QuanticsTCI.jl` are adopted and re-exported through
wrapper modules, but they are not owned by `Tensor4all.jl`.

## Layers

| Module | Responsibility | Current state |
|------|----------------|---------------|
| `Core` | `Index`, `Tensor`, base metadata behavior | implemented |
| `TensorNetworks` | indexed chain wrapper, helper surface, operator boundary | implemented as public chain layer |
| `ITensorCompat` | opt-in ITensors/ITensorMPS migration facade over `TensorNetworks.TensorTrain` | implemented |
| `SimpleTT` | raw-array tensor trains, compression, MPO contraction | implemented |
| `TensorCI` | interpolation boundary returning `TensorCI2` | implemented as adapter layer |
| `QuanticsGrids` | adopted grid re-export layer | implemented |
| `QuanticsTCI` | adopted quantics-TCI re-export layer | implemented |
| `QuanticsTransform` | quantics-specific constructors of `TensorNetworks.LinearOperator` | partially implemented |
| HDF5 extension | pure Julia `save_as_mps` / `load_tt` | implemented |

## Key Boundaries

- `TensorCI` should not return indexed `TensorNetworks.TensorTrain`.
- `TensorCI.crossinterpolate2` returns `TensorCI2`; `SimpleTT` owns conversion
  into raw-array TT form.
- `SimpleTT` owns raw-array numerics.
- `TensorNetworks` adds index semantics, chain helpers, `LinearOperator`,
  `apply`, and HDF5 interoperability.
- `ITensorCompat` provides source-compatibility wrappers such as `MPS` and
  `MPO`; it does not replace `TensorNetworks.TensorTrain`.
- The broader chain-helper surface in `TensorNetworks` is implemented in pure
  Julia.
- `ITensorCompat` forwards to `TensorNetworks`; it is cutoff-only for
  truncation and does not replace the native `threshold` / `svd_policy`
  controls.
- Operator-space setters are explicit `Vector{Index}` APIs rather than
  TensorTrain-driven auto-binding.
- The Julia-facing C API target is reduced and chain-oriented.

## Two TensorTrain Types

The codebase has two separate tensor-train types:

- **`TensorNetworks.TensorTrain`** stores `Vector{Tensor}` with index metadata.
  Use this for index-aware operations: site queries, index replacement, HDF5
  save/load, and operator application.
- **`SimpleTT.TensorTrain{T,N}`** stores `Vector{Array{T,N}}` as raw arrays.
  Use this for numerical algorithms: compression, MPO-MPO contraction, and as
  the output of `TensorCI.crossinterpolate2`.

There is currently no automatic conversion between the two types. They serve
different purposes in the module hierarchy: `SimpleTT` handles numerics,
`TensorNetworks` handles indexed semantics.

`ITensorCompat.MPS` and `ITensorCompat.MPO` wrap `TensorNetworks.TensorTrain`
for migration-oriented workflows. Raw MPS blocks use `(left_link, site,
right_link)` order; raw MPO blocks use `(left_link, input_site, output_site,
right_link)` order.

## Still Deferred

- deeper `QuanticsTransform` kernel coverage and edge-case validation
- broader non-chain behavior
- any remaining reduced `tensor4all-rs` ABI documentation gaps
