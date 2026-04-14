# Design Documents

This directory contains the restored Julia frontend design set for
`Tensor4all.jl`.

The goal is to document the old public split explicitly:

- `Core`
- `TensorNetworks`
- `SimpleTT`
- `TensorCI`
- `QuanticsTransform`

The current implementation is still smaller than the full historical scope, but
the docs now describe the restored Julia-side architecture rather than the
temporary TreeTN-first skeleton.

## Entry Points

- [julia_ffi.md](./julia_ffi.md) for the overall index
- [julia_ffi_core.md](./julia_ffi_core.md) for low-level primitives and the reduced C API assumption
- [julia_ffi_tensornetworks.md](./julia_ffi_tensornetworks.md) for `TensorNetworks.TensorTrain`
- [julia_ffi_simplett.md](./julia_ffi_simplett.md) for `SimpleTT.TensorTrain{T,N}`
- [julia_ffi_tci.md](./julia_ffi_tci.md) for `TensorCI -> SimpleTT`
- [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) for `QuanticsTransform`

## Compatibility Notes

- HDF5 compatibility uses `save_as_mps` / `load_tt` and stores data using the
  `MPS` schema.
- The Julia-side docs assume a minimized, chain-oriented C API target.
- `TreeTensorNetwork` remains in the repository, but it is not the primary
  public architecture for this phase.

## Deferred Material

The follow-up implementation plan remains in
[../plans/2026-04-10-tensor4all-rework-followup.md](../plans/2026-04-10-tensor4all-rework-followup.md).
