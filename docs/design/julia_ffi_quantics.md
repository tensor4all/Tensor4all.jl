# Julia Frontend Quantics Layer

## Purpose

This document covers the quantics layer on the Julia side: adopted grid functionality from `QuanticsGrids.jl`, plus `Tensor4all.jl`-owned quantics-specific transform and backend-integration behavior.

## In Scope

- adoption and re-export of `QuanticsGrids.jl` grid and coordinate-conversion APIs
- documentation of supported layout and index-table conventions inherited from `QuanticsGrids.jl`
- `Tensor4all.jl`-owned quantics transform semantics
- backend-facing quantics integration points
- multiresolution expectations for higher layers where they depend on the quantics layer

This document does not cover `TTFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## Ownership and Re-Export Boundary

- `QuanticsGrids.jl` owns quantics grid types, layout/index-table semantics, and coordinate-conversion behavior.
- `Tensor4all.jl` should adopt and re-export a reviewed subset of that surface so users can access it through a single import.
- `Tensor4all.jl` owns quantics transform constructors, backend operator materialization, and the integration between the adopted grid layer and the TT / TreeTN layer.
- Re-export improves usability but does not transfer ownership.

## Grid Semantics

- Grid variables should remain named and inspectable through the adopted `QuanticsGrids.jl` interface.
- Layout and index-table behavior should follow `QuanticsGrids.jl` conventions rather than being redefined locally.
- Coordinate conversion should be available to `Tensor4all.jl` users through the re-exported `QuanticsGrids.jl` layer.
- Endpoint conventions and mixed bases should likewise be inherited from `QuanticsGrids.jl` and documented clearly.

## Layout Conventions

- fused layouts for compact representations
- interleaved layouts for variable/bit interleaving
- grouped layouts for user-defined groupings
- explicit index-table control when a specific unfolding is needed

## Transform Semantics

Quantics transforms are owned by `Tensor4all.jl` and can be represented as backend operators exposed through Julia:

- affine pullbacks
- shifts
- flips and reversals
- phase rotation
- cumulative sums
- Fourier-style transforms
- binary operations on two variables

## Multiresolution

- coarsening and averaging
- interpolation and refinement
- layout-preserving embed/resample workflows

## Relationship to Other Docs

- [julia_ffi_tt.md](./julia_ffi_tt.md) covers the backend TT operator layer that can carry these transforms.
- [bubbleteaCI.md](./bubbleteaCI.md) covers the higher-level function workflows that consume quantics grids.

## Open Questions

- Should `Tensor4all.jl` re-export the full `QuanticsGrids.jl` public surface or start with a curated subset?
- Which `QuanticsGrids.jl` conventions should be documented as part of the reviewed `Tensor4all.jl` public story from day one?
- Where should weighted integration and quadrature-like behavior live?
