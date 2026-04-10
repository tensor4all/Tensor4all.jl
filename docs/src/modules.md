# Architecture Status

## Current Phase

`Tensor4all.jl` is in a review-first skeleton phase.

The package now exposes real metadata-level types and integration boundaries for
review, while backend numerics remain intentionally stubbed.

## Current Layers

| Layer | Responsibility | Current status |
|------|----------------|----------------|
| Core | lazy backend loading, common errors, `Index`, `Tensor` | metadata behavior implemented |
| TreeTN | `TreeTensorNetwork`, chain aliases, runtime topology predicates | metadata behavior implemented |
| Quantics reuse | adopted `QuanticsGrids.jl` grid and coordinate-conversion surface | curated re-export implemented |
| Quantics local layer | transform descriptors and QTCI placeholders | metadata/stub behavior implemented |
| Extensions | ITensors and HDF5 compatibility glue | extension-only stubs implemented |
| BubbleTeaCI | `TTFunction` and high-level function workflows | intentionally out of scope here |

## Behavior Boundary

- Metadata constructors, inspection helpers, and topology predicates work.
- Backend-backed operations such as contraction, dense materialization, and
  transforms deliberately throw `SkeletonNotImplemented`.
- Importing `Tensor4all.jl` does not require a compiled `tensor4all-rs` backend.

## Ownership and Re-Export

- `tensor4all-rs` owns kernels, storage, and numerically heavy backend behavior.
- `Tensor4all.jl` owns Julia-side wrappers, TreeTN-general abstractions, local
  quantics transforms, QTCI placeholders, and compatibility extensions.
- `QuanticsGrids.jl` owns quantics grid semantics and coordinate conversion.
- `Tensor4all.jl` re-exports a curated `QuanticsGrids.jl` surface for usability,
  but that re-export does not change ownership.
- `BubbleTeaCI` remains the home of `TTFunction` and high-level function
  workflows, and should consume lower-level functionality from `Tensor4all.jl`
  and adopted dependencies instead of duplicating it.

## Review Questions Still Open

- Should `Index` and `Tensor` gain explicit backend-handle behavior beyond the
  current nullable-handle skeleton fields before backend integration starts?
- Is the current curated `QuanticsGrids.jl` re-export scope broad enough for the
  first downstream consumers?
- Is the downstream `BubbleTeaCI` contract explicit enough before migration
  begins?
- Are the current public names good enough to freeze for backend implementation?
