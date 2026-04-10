# Julia Frontend Quantics Layer

## Purpose

This document covers quantics grid semantics and quantics-specific transform behavior on the Julia side.

## In Scope

- named-variable quantics grids
- site layouts and index-table control
- coordinate conversion
- layout conventions such as fused, interleaved, and grouped representations
- quantics transform semantics
- multiresolution support

This document does not cover `TTFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## Grid Semantics

- Grid variables should be named, not just positional.
- Layout should be explicit and inspectable.
- Coordinate conversion should be available at the Julia level.
- Endpoint conventions and mixed bases should be represented clearly.

## Layout Conventions

- fused layouts for compact representations
- interleaved layouts for variable/bit interleaving
- grouped layouts for user-defined groupings
- explicit index-table control when a specific unfolding is needed

## Transform Semantics

Quantics transforms can be represented as backend operators and exposed through Julia:

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

- Which layout should be canonical internally?
- Which layout forms should be accepted as user-facing construction syntax?
- Where should weighted integration and quadrature-like behavior live?
