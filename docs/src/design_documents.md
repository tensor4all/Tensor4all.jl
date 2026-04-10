# Design Documents

The active architecture discussion lives in the repository design set under
`docs/design/`.

## Main Entry Points

- [Design index on GitHub](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/README.md)
- [Julia frontend overview](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/julia_ffi.md)
- [Core primitives](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/julia_ffi_core.md)
- [TreeTN and chain support](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/julia_ffi_tt.md)
- [Quantics layer](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/julia_ffi_quantics.md)
- [BubbleTeaCI migration boundary](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/bubbleteaCI.md)
- [Extensions and compatibility](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/julia_ffi_extensions.md)
- [Roadmap](https://github.com/tensor4all/Tensor4all.jl/blob/tensor4all-rework/docs/design/julia_ffi_roadmap.md)

## How To Read Them

Start with the overview, then the TreeTN and quantics documents, then the
roadmap. The `BubbleTeaCI` note is important for clarifying what does not belong
in `Tensor4all.jl`.

Two cross-package boundaries are especially important while reviewing the design:

- `QuanticsGrids.jl` owns grid semantics and coordinate conversion; `Tensor4all.jl`
  adopts and re-exports a curated subset.
- `BubbleTeaCI` owns `TTFunction` and high-level workflows; it should build on
  `Tensor4all.jl` rather than duplicating lower layers.

For the implementation work intentionally staged beyond the initial reset, see
[Deferred Rework Plan](deferred_rework_plan.md).
