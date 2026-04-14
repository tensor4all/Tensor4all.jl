# Julia Frontend QuanticsTransform Layer

## Purpose

This document covers the Julia-side operator boundary that sits on top of the
restored TT layers.

## In Scope

- `QuanticsTransform.LinearOperator`
- operator materialization and apply orchestration
- the reduced chain-oriented C API assumption for transform kernels

This document does not define the raw TT numerics or the chain container.

## Public Shape

```julia
struct LinearOperator
    payload
end
```

- The public Julia type is intentionally lightweight.
- The payload carries the operator description or backend handle needed by the
  higher-level API.
- The real numeric work is expected to be delegated to reusable kernels.

## Kernel Boundary

The docs for this phase assume a minimized, chain-oriented backend ABI:

- materialization kernels for TT-like operators
- apply kernels for chain subsets
- no tree-specific public ABI promises

That keeps `QuanticsTransform` aligned with the reduced Julia frontend story and
avoids reintroducing a TreeTN-first public contract.

## Relationship to the TT Layers

- `TensorNetworks` provides the public chain container.
- `SimpleTT` provides raw-array TT numerics.
- `TensorCI` provides interpolation output in `SimpleTT` form.
- `QuanticsTransform` sits on top of those layers as an operator boundary.

## Open Questions

- Which operator semantics should remain Julia-owned versus delegated to Rust
  kernels?
- How much layout information should be explicit in Julia, given the reduced
  C API assumption?
