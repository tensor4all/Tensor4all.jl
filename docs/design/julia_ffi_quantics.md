# Julia Frontend Quantics Layer

## Purpose

This document covers the adopted quantics-facing modules on the Julia side:
`QuanticsGrids`, `QuanticsTCI`, and the relationship between those wrappers and
`QuanticsTransform`.

## In Scope

- adoption and re-export of `QuanticsGrids.jl`
- adoption and re-export of `QuanticsTCI.jl`
- the ownership split between these wrappers and `QuanticsTransform`
- backend-facing quantics integration points

This document does not cover `TTFunction` semantics. Those remain in [bubbleteaCI.md](./bubbleteaCI.md).

## Ownership and Re-Export Boundary

- `QuanticsGrids.jl` owns quantics grid types, layout/index-table semantics,
  and coordinate-conversion behavior.
- `QuanticsTCI.jl` owns its quantics-TCI algorithms and convenience functions.
- `Tensor4all.jl` re-exports both through wrapper submodules so users can stay
  in a single import.
- `Tensor4all.jl` owns quantics transform constructors, while
  `TensorNetworks` owns the generic `LinearOperator` type and `apply`.
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

## Relationship to `QuanticsTransform`

- `QuanticsTransform` constructs quantics-specific
  `TensorNetworks.LinearOperator` values.
- `QuanticsGrids` and `QuanticsTCI` do not own `LinearOperator`.
- Operator application belongs in `TensorNetworks`, not in the quantics wrapper
  modules.

## Relationship to Other Docs

- [julia_ffi_tensornetworks.md](./julia_ffi_tensornetworks.md) covers the
  generic chain and operator layer.
- [julia_ffi_quanticstransform.md](./julia_ffi_quanticstransform.md) covers the
  quantics-specific operator constructors.
- [bubbleteaCI.md](./bubbleteaCI.md) covers the higher-level function workflows
  that consume quantics grids.

## Open Questions

- Which helper functions from the historical `Quantics.jl` layer should be
  restored in Phase 2?
- Which `QuanticsTCI.jl` conveniences should be documented as part of the
  single-import `Tensor4all.jl` story?
