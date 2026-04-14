# Module Overview

## Current Architecture

The restored Julia frontend is layered like this:

```text
Core (Index, Tensor)
  ↓
TensorNetworks (public chain wrapper, HDF5 boundary)
  ↓
SimpleTT (raw-array TT numerics)
  ↓
TensorCI (interpolation boundary back into SimpleTT)
  ↓
QuanticsTransform (operator boundary)
```

`QuanticsGrids.jl` is adopted and re-exported for grid semantics, but it is not
owned by `Tensor4all.jl`.

## Layers

| Module | Responsibility | Current state |
|------|----------------|---------------|
| `Core` | `Index`, `Tensor`, base metadata behavior | implemented |
| `TensorNetworks` | indexed chain wrapper with `data`, `llim`, `rlim` | implemented as public chain type |
| `SimpleTT` | raw-array tensor trains, compression, MPO contraction | implemented in POC scope |
| `TensorCI` | interpolation boundary returning `SimpleTT` | implemented as POC adapter |
| `QuanticsTransform` | Julia-owned operator semantics | skeleton / deferred |
| HDF5 extension | pure Julia `save_as_mps` / `load_tt` | implemented |

## Key Boundaries

- `TensorCI` should not return indexed `TensorNetworks.TensorTrain`.
- `SimpleTT` owns raw-array numerics.
- `TensorNetworks` adds index semantics and HDF5 interoperability.
- The Julia-facing C API target is reduced and chain-oriented.

## Still Deferred

- finalized reduced `tensor4all-rs` ABI documentation
- deeper `QuanticsTransform` kernel coverage
- broader non-chain / TreeTN behavior
