# Design Documents

The active Julia frontend design set lives in `docs/design/`.

## Main Entry Points

- [Design index on GitHub](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/README.md)
- [Julia frontend overview](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/julia_ffi.md)
- [Core primitives](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/julia_ffi_core.md)
- [TensorNetworks layer](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/julia_ffi_tensornetworks.md)
- [SimpleTT layer](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/julia_ffi_simplett.md)
- [TensorCI layer](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/julia_ffi_tci.md)
- [QuanticsTransform layer](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/design/julia_ffi_quanticstransform.md)

## Reading Order

Start with the overview, then read `Core`, `TensorNetworks`, `SimpleTT`, and
`TensorCI` in that order. `QuanticsTransform` describes the deferred operator
story that sits on top of those layers.

The single implementation handoff plan lives in
[2026-04-10 tensor4all rework follow-up plan](https://github.com/tensor4all/Tensor4all.jl/blob/main/docs/plans/2026-04-10-tensor4all-rework-followup.md).
