# Julia Frontend TensorCI Boundary

## Purpose

This document defines the interpolation boundary between function approximation
and the raw numerical TT layer.

## In Scope

- `TensorCI.crossinterpolate2`
- the `TensorCI -> SimpleTT` handoff
- small adapters that normalize interpolation output into `SimpleTT.TensorTrain`

This document does not define the interpolation algorithm itself. It only
describes the Julia-facing boundary and the resulting type shape.

## Public Contract

```julia
crossinterpolate2(::Type{T}, f, localdims; kwargs...)::SimpleTT.TensorTrain{T,3}
```

- `localdims` describes the local domain sizes.
- For multi-site domains, the implementation delegates to
  `TensorCrossInterpolation.crossinterpolate2(...)` and adapts the result.
- For the required one-site fallback, the boundary stays local because the
  upstream package rejects a 1D domain in that path.

## Architectural Role

- `TensorCI` does not own chain topology or indexed tensor semantics.
- `TensorCI` does not own raw-array TT compression.
- `TensorCI` exists to return `SimpleTT` output in the shape that later layers
  already understand.

## Boundary Rules

- Do not widen this layer into a second TT toolkit.
- Keep the public surface small and focused on interpolation output.
- Preserve the `SimpleTT` return type even when the source algorithm is
  delegated to another package.

## Open Questions

- Which interpolation variants, if any, should be surfaced here in addition to
  `crossinterpolate2`?
- Should the one-site fallback remain local, or should upstream support be
  pushed until it can be delegated too?
