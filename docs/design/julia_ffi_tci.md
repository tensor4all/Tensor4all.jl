# Julia Frontend TensorCI Boundary

## Purpose

This document defines the interpolation boundary between function approximation
and the raw numerical TT layer.

## In Scope

- `TensorCI.crossinterpolate2`
- `TensorCI2` as the public return type
- the `SimpleTT.TensorTrain(tci)` conversion handoff
- re-export of the upstream `TensorCrossInterpolation.jl` surface where practical

This document does not define the interpolation algorithm itself. It only
describes the Julia-facing boundary and the resulting type shape.

## Public Contract

```julia
crossinterpolate2(::Type{T}, f, localdims; kwargs...)::TensorCI2{T}
```

- `localdims` describes the local domain sizes.
- For multi-site domains, the implementation delegates to
  `TensorCrossInterpolation.crossinterpolate2(...)` and returns the resulting
  `TensorCI2`.
- The current Phase 1 skeleton requires `length(localdims) >= 2`.
- Conversion into raw-array TT form happens through
  `SimpleTT.TensorTrain(tci)`.

## Architectural Role

- `TensorCI` does not own chain topology or indexed tensor semantics.
- `TensorCI` does not own raw-array TT compression.
- `TensorCI` exists to expose interpolation results and the upstream API shape
  without becoming a second TT toolkit.

## Boundary Rules

- Do not widen this layer into a second TT toolkit.
- Keep the public surface focused on interpolation output and upstream
  compatibility.
- Do not make `TensorCI` return indexed `TensorNetworks.TensorTrain`.

## Open Questions

- Which interpolation variants, if any, should be surfaced here in addition to
  `crossinterpolate2`?
- Should one-site `TensorCI2` construction be added locally, or should that
  remain an upstream request?
