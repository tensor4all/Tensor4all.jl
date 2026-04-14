# Julia Frontend Design for `tensor4all-rs`

## Overview

This design set documents the restored Julia frontend architecture:

- `Core` owns `Index`, `Tensor`, backend loading, and error handling.
- `TensorNetworks` owns the public chain container `TensorTrain = Vector{Tensor} + llim/rlim`.
- `TensorNetworks` also owns `LinearOperator` and `apply`.
- `SimpleTT` owns the raw-array TT numerics.
- `TensorCI` returns `TensorCI2` and re-exports the upstream interpolation surface.
- `QuanticsGrids` and `QuanticsTCI` are adopted wrapper re-export modules.
- `QuanticsTransform` owns quantics-specific operator constructors.

The docs here assume a reduced, chain-oriented C API target on the Rust side.

## Doc Map

| File | Purpose |
|------|---------|
| [julia_ffi_core.md](./julia_ffi_core.md) | `Index`, `Tensor`, error handling, and reduced C API assumptions |
| [julia_ffi_tensornetworks.md](./julia_ffi_tensornetworks.md) | `TensorNetworks.TensorTrain`, `LinearOperator`, and `apply` |
| [julia_ffi_simplett.md](./julia_ffi_simplett.md) | `SimpleTT.TensorTrain{T,N}` and pure Julia TT numerics |
| [julia_ffi_tci.md](./julia_ffi_tci.md) | `TensorCI2` return boundary and `SimpleTT` conversion |
| [julia_ffi_quantics.md](./julia_ffi_quantics.md) | adopted `QuanticsGrids` / `QuanticsTCI` wrappers |
| [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) | `QuanticsTransform` constructors and Rust kernel boundary |

## Ownership Model

- `tensor4all-rs` owns performance-critical kernels, storage, and numerics.
- The Julia frontend owns the public module split, validation, composition, and
  compatibility glue.
- `save_as_mps` / `load_tt` live at the HDF5 extension boundary and use the
  `MPS` schema.

## Reuse Boundary

- When a focused Julia package already owns a reusable concept cleanly, prefer
  adopting it over reimplementing it.
- The Julia docs still assume `QuanticsGrids.jl` owns grid semantics and
  coordinate conversion.
- Re-export is a usability choice for single-import workflows. It does not
  change ownership of the underlying functionality.

## Cross-Cutting Questions

- Which chain-oriented primitives remain in Julia, and which ones stay in Rust
  for the minimized C API?
- How much of the TT contraction/compression surface should be expressed in
  Julia versus exposed as reusable backend kernels?
- How should the HDF5 boundary preserve ITensorMPS compatibility while keeping
  the Julia-side API focused on `TensorNetworks.TensorTrain`?
